from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from experiments.daisy_cable_replanning.trials import (
    CableDemoExperiment,
    CableDemoTrial,
    Command,
    TrialFiles,
    TrialOption,
)

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
        case Command.TRIAL:
            _trial(arguments).run()


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
    record.set_defaults(illegal_joint_states=None)
    replan.add_argument("--illegal-joint-states", type=Path, required=True)

    trial = commands.add_parser(Command.TRIAL, help="run the demo once")
    trial.add_argument(TrialOption.NUMBER, type=int, required=True)
    trial.add_argument(TrialOption.SEED, type=int, required=True)
    trial.add_argument(TrialOption.DIRECTORY, type=Path, required=True)
    trial.add_argument(TrialOption.ILLEGAL_JOINT_STATES, type=Path, default=None)
    trial.add_argument(TrialOption.VIDEO, type=Path, default=None)
    return parser
