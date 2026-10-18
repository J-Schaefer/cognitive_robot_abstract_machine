from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from experiments.daisy_cable_replanning.summary import (
    ExperimentOutcome,
    ReplanningComparison,
)
from experiments.daisy_cable_replanning.trials import (
    CableDemoExperiment,
    CableDemoTrial,
    Command,
    TrialFiles,
    TrialOption,
)
from experiments.daisy_cable_replanning.video import ReplanningVideo

# %% command line


def run_from_command_line() -> None:
    """
    Run the command the command line names.
    """
    arguments = _parser().parse_args()
    match arguments.command:
        case Command.RECORD | Command.REPLAN:
            CableDemoExperiment(
                output_directory=arguments.output,
                trials=arguments.trials,
                first_seed=arguments.first_seed,
                workers=arguments.workers,
                illegal_joint_states=arguments.illegal_joint_states,
            ).run()
        case Command.SUMMARIZE:
            _summarize(arguments)
        case Command.VIDEO:
            print(_video(arguments).record())
        case Command.TRIAL:
            _trial(arguments).run()


def _summarize(arguments: Namespace) -> None:
    """
    Print the summary of the experiment, and its comparison with the baseline if one is
    given.
    """
    outcome = ExperimentOutcome.load(arguments.output)
    print(outcome.describe())
    if arguments.baseline is not None:
        print("\ncompared with the baseline:")
        print(
            ReplanningComparison(
                baseline=ExperimentOutcome.load(arguments.baseline),
                replanning=outcome,
            ).describe()
        )


def _video(arguments: Namespace) -> ReplanningVideo:
    """
    :return: The comparison video the arguments of the :attr:`Command.VIDEO` command
        describe.
    """
    if arguments.seed is not None:
        return ReplanningVideo(
            output_directory=arguments.output,
            seed=arguments.seed,
            illegal_joint_states=arguments.illegal_joint_states,
        )
    return ReplanningVideo.of_first_improved_seed(
        output_directory=arguments.output,
        baseline=ExperimentOutcome.load(arguments.baseline),
        replanning=ExperimentOutcome.load(arguments.replanning),
        illegal_joint_states=arguments.illegal_joint_states,
    )


def _trial(arguments: Namespace) -> CableDemoTrial:
    """
    :return: The trial the arguments of the :attr:`Command.TRIAL` command describe.
    """
    return CableDemoTrial(
        number=arguments.number,
        seed=arguments.seed,
        files=TrialFiles(directory=arguments.directory, number=arguments.number),
        illegal_joint_states=arguments.illegal_joint_states,
        video=arguments.video,
    )


def _parser() -> ArgumentParser:
    """
    :return: The parser of the command line.
    """
    parser = ArgumentParser(
        description="Repeated runs of the DAiSy cable demo, recording illegal joint "
        "states and replanning around them."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    record = commands.add_parser(Command.RECORD, help="record illegal joint states")
    replan = commands.add_parser(
        Command.REPLAN, help="replan around recorded illegal joint states"
    )
    for experiment in (record, replan):
        experiment.add_argument("--output", type=Path, required=True)
        experiment.add_argument("--trials", type=int, default=100)
        experiment.add_argument("--first-seed", type=int, default=0)
        experiment.add_argument("--workers", type=int, default=2)
    record.set_defaults(illegal_joint_states=[])
    replan.add_argument(
        TrialOption.ILLEGAL_JOINT_STATES, type=Path, nargs="+", required=True
    )

    summarize = commands.add_parser(
        Command.SUMMARIZE, help="summarize a finished experiment"
    )
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="output of the experiment of the same seeds without replanning",
    )

    video = commands.add_parser(
        Command.VIDEO, help="record one seed with and without replanning"
    )
    video.add_argument("--output", type=Path, required=True)
    video.add_argument(
        TrialOption.ILLEGAL_JOINT_STATES, type=Path, nargs="+", required=True
    )
    video.add_argument(
        TrialOption.SEED,
        type=int,
        default=None,
        help="seed to record; by default the first seed that collided in --baseline "
        "and completed without any collision in --replanning",
    )
    video.add_argument("--baseline", type=Path, default=None)
    video.add_argument("--replanning", type=Path, default=None)

    trial = commands.add_parser(Command.TRIAL, help="run the demo once")
    trial.add_argument(TrialOption.NUMBER, type=int, required=True)
    trial.add_argument(TrialOption.SEED, type=int, required=True)
    trial.add_argument(TrialOption.DIRECTORY, type=Path, required=True)
    trial.add_argument(
        TrialOption.ILLEGAL_JOINT_STATES, type=Path, nargs="+", default=[]
    )
    trial.add_argument(TrialOption.VIDEO, type=Path, default=None)
    return parser
