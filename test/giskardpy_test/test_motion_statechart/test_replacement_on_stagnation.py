"""
Replacing a motion that has stopped approaching its goal short of reaching it.
"""

from __future__ import annotations

from datetime import timedelta

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.progress_monitors import NoLongerProgressing
from giskardpy.motion_statechart.monitors.templates import ReplacedWhenTrue
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    ConstTrueNode,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from semantic_digital_twin.world import World

# %% helpers


def _replaced_when_stuck(world: World, held_back: bool) -> ReplacedWhenTrue:
    """
    Run a motion of the world's only connection towards 0.9, replaced once it stops
    progressing; a stronger task holding the connection at zero keeps it from getting
    there if it is held back.

    :return: The goal replacing the motion, after the chart ended.
    """
    (connection,) = world.connections
    motion = JointPositionList(
        goal_state=JointState.from_mapping({connection: 0.9}), name="motion"
    )
    goal = ReplacedWhenTrue(
        monitor=NoLongerProgressing(
            monitored_node=motion, timeout=timedelta(seconds=0.5)
        ),
        monitored_node=motion,
        replacement=ConstTrueNode(name="replacement"),
    )
    goal.end_condition = goal.observation_variable
    msc = MotionStatechart()
    msc.add_node(goal)
    if held_back:
        msc.add_node(
            JointPositionList(
                goal_state=JointState.from_mapping({connection: 0.0}),
                name="holding back",
                weight=motion.weight * 100,
            )
        )
    end = EndMotion()
    end.start_condition = goal.is_terminated
    msc.add_node(end)
    executor = Executor(MotionStatechartContext(world=world))
    executor.compile(motion_statechart=msc)
    executor.tick_until_end(timeout=500)
    return goal


# %% stagnation


def test_a_motion_held_back_from_its_goal_is_replaced(prismatic_bot: World):
    goal = _replaced_when_stuck(prismatic_bot, held_back=True)

    assert goal.monitored_node.life_cycle_state == LifeCycleValues.FAILED
    assert goal.replacement.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert goal.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_motion_that_progresses_is_not_replaced(prismatic_bot: World):
    goal = _replaced_when_stuck(prismatic_bot, held_back=False)

    assert goal.monitored_node.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert goal.replacement.life_cycle_state == LifeCycleValues.NOT_STARTED
