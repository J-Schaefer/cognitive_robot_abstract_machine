"""
Stopping a motion that has stopped approaching its goal short of reaching it, and
running another node in its place.
"""

from __future__ import annotations

from datetime import timedelta

from cramph.composites import StoppedWhenTrue, TryInOrder
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues
from cramph.executor import StatechartExecutor
from cramph.nodes_for_testing import ConstTrueNode
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.progress_monitors import Stalled
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from semantic_digital_twin.world import World

# %% helpers


def _replaced_when_stuck(world: World, held_back: bool) -> TryInOrder:
    """
    Run a motion of the world's only connection towards 0.9, replaced once it stops
    progressing; a stronger task holding the connection at zero keeps it from getting
    there if it is held back.

    :return: The goal stopping the motion and running the replacement after it, once the
        chart ended.
    """
    (connection,) = world.connections
    motion = JointPositionList(
        goal_state=JointState.from_mapping({connection: 0.9}), name="motion"
    )
    goal = TryInOrder(
        nodes=[
            StoppedWhenTrue(
                monitor=Stalled(monitored_node=motion, timeout=timedelta(seconds=0.5)),
                monitored_node=motion,
            ),
            ConstTrueNode(name="replacement"),
        ]
    )
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(node=goal)
    if held_back:
        statechart.add_node(
            node=JointPositionList(
                goal_state=JointState.from_mapping({connection: 0.0}),
                name="holding back",
                weight=motion.weight * 100,
            )
        )
    end = EndMotion()
    end.start_condition = goal.is_terminated
    statechart.add_node(node=end)
    executor.compile(statechart=statechart)
    executor.tick_until_end(timeout=500)
    return goal


# %% stagnation


def test_a_motion_held_back_from_its_goal_is_replaced(prismatic_bot: World):
    goal = _replaced_when_stuck(prismatic_bot, held_back=True)
    stopped_motion, replacement = goal.nodes

    assert stopped_motion.monitored_node.life_cycle_state == LifeCycleValues.INTERRUPTED
    assert replacement.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert goal.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_motion_that_progresses_is_not_replaced(prismatic_bot: World):
    goal = _replaced_when_stuck(prismatic_bot, held_back=False)

    assert goal.life_cycle_state == LifeCycleValues.SUCCEEDED
    _, replacement = goal.nodes

    assert replacement.life_cycle_state == LifeCycleValues.NOT_STARTED
