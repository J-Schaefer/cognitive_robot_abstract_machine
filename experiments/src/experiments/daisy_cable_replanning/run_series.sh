#!/usr/bin/env bash
# Runs one series of the DAiSy cable replanning experiment end to end; see README.md.
#
#   run_series.sh <output directory> [replanning rounds] [extra databases...]
#
# 1. record:          training seeds 0-99, the demo as it is
# 2. heldout_record:  held-out seeds 100-199, the demo as it is (the baseline)
# 3. replan_round_N:  training seeds, replanning around every database so far; the
#                     illegal joint states each round still runs into are added
# 4. heldout_replan:  held-out seeds, replanning around all databases
# 5. video:           the first held-out seed that collided in the baseline, with and
#                     without replanning, side by side
#
# Extra databases (e.g. of an earlier series) are used in every replanning step.
set -euo pipefail

OUTPUT=${1:?output directory}
ROUNDS=${2:-3}
shift $(( $# < 2 ? $# : 2 ))
EXTRA_DATABASES=("$@")

TRIALS=${TRIALS:-100}
WORKERS=${WORKERS:-6}
MEMORY=${MEMORY:-9G}
HELDOUT_SEED=${HELDOUT_SEED:-100}

# Caps the memory of a whole run; ulimit -v would break the video's OpenGL context.
capped() { systemd-run --user --scope -q -p MemoryMax="$MEMORY" -p MemorySwapMax=0 "$@"; }
experiment() { capped python -m experiments.daisy_cable_replanning "$@"; }

mkdir -p "$OUTPUT"
experiment record --output "$OUTPUT/record" --first-seed 0 --trials "$TRIALS" \
    --workers "$WORKERS" > "$OUTPUT/record.log" 2>&1
experiment record --output "$OUTPUT/heldout_record" --first-seed "$HELDOUT_SEED" \
    --trials "$TRIALS" --workers "$WORKERS" > "$OUTPUT/heldout_record.log" 2>&1

DATABASES=("${EXTRA_DATABASES[@]}" "$OUTPUT/record/illegal_joint_states.json")
for ROUND in $(seq 1 "$ROUNDS"); do
    experiment replan --output "$OUTPUT/replan_round_$ROUND" --first-seed 0 \
        --trials "$TRIALS" --workers "$WORKERS" \
        --illegal-joint-states "${DATABASES[@]}" > "$OUTPUT/replan_round_$ROUND.log" 2>&1
    DATABASES+=("$OUTPUT/replan_round_$ROUND/illegal_joint_states.json")
done

experiment replan --output "$OUTPUT/heldout_replan" --first-seed "$HELDOUT_SEED" \
    --trials "$TRIALS" --workers "$WORKERS" \
    --illegal-joint-states "${DATABASES[@]}" > "$OUTPUT/heldout_replan.log" 2>&1

for RUN in record heldout_record $(seq -f "replan_round_%g" 1 "$ROUNDS") heldout_replan; do
    echo "== $RUN"
    BASELINE=$([[ $RUN == heldout_replan ]] && echo "$OUTPUT/heldout_record" \
        || ([[ $RUN == replan_round_* ]] && echo "$OUTPUT/record") || true)
    python -m experiments.daisy_cable_replanning summarize --output "$OUTPUT/$RUN" \
        ${BASELINE:+--baseline "$BASELINE"}
done > "$OUTPUT/summary.txt"

experiment video --output "$OUTPUT/video" --baseline "$OUTPUT/heldout_record" \
    --replanning "$OUTPUT/heldout_replan" --illegal-joint-states "${DATABASES[@]}" \
    > "$OUTPUT/video.log" 2>&1
