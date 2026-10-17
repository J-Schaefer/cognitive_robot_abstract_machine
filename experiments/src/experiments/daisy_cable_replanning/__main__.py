"""
Command line entry point of the DAiSy cable replanning experiment::

    python3 -m experiments.daisy_cable_replanning record --output results/record
    python3 -m experiments.daisy_cable_replanning replan --output results/replan \
        --illegal-joint-states results/record/illegal_joint_states.json
"""

from experiments.daisy_cable_replanning.command_line import run_from_command_line

if __name__ == "__main__":
    run_from_command_line()
