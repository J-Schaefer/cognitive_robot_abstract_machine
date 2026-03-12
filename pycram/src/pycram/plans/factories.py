from __future__ import annotations
from typing_extensions import List, assert_never, Union, Optional, TYPE_CHECKING

from krrood.entity_query_language.query.match import Match
from pycram.datastructures.dataclasses import Context

from pycram.plans.plan import Plan


if TYPE_CHECKING:
    from pycram.language import SequentialNode
    from pycram.plans.plan_node import ActionLike, PlanNode


def execute_single(
    action_like: ActionLike, context: Optional[Context] = None
) -> PlanNode:

    node = make_node(action_like)
    plan = Plan(context=context)
    plan.add_node(node)
    return node


def sequential(
    children: List[ActionLike],
    context: Optional[Context] = None,
) -> SequentialNode:
    from pycram.language import SequentialNode

    root = SequentialNode()
    plan = Plan(context=context)
    plan.add_node(root)
    for action_like in children:
        plan.add_edge(root, make_node(action_like))
    return root


def make_node(action_like: ActionLike) -> PlanNode:
    from pycram.plans.plan_node import (
        PlanNode,
        UnderspecifiedActionNode,
        ActionNode,
        MotionNode,
    )
    from pycram.robot_plans.actions.base import ActionDescription
    from pycram.robot_plans import BaseMotion

    if isinstance(action_like, PlanNode):
        return action_like
    elif isinstance(action_like, Match):
        underspecified_action = UnderspecifiedActionNode(
            underspecified_action=action_like
        )
        return underspecified_action
    elif isinstance(action_like, ActionDescription):
        return ActionNode(designator=action_like)
    elif isinstance(action_like, BaseMotion):
        return MotionNode(designator=action_like)
    else:
        assert_never(action_like)
