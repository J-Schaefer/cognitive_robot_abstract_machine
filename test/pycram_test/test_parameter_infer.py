from dataclasses import dataclass

from typing_extensions import List

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import ApproachDirection, VerticalAlignment, Arms
from pycram.datastructures.grasp import GraspDescription
from pycram.language import SequentialPlan
from pycram.parameter_inference import (
    ParameterInferenceRule,
    Domain,
    ParameterIdentifier,
)
from pycram.parameter_rules.default_type_rules import EnumDomain, GraspDomain
from pycram.plan import Plan
from pycram.robot_plans import PickUpActionDescription


@dataclass
class DummyRule(ParameterInferenceRule):

    n_th: int

    def apply(self, domain: List, context: Context) -> List:
        return domain[: self.n_th]


def test_infer_enum_domain(immutable_model_world):

    world, robot_view, context = immutable_model_world

    pick_action = PickUpActionDescription(
        world.get_body_by_name("milk.stl"),
        ...,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            robot_view.right_arm.manipulator,
        ),
    )

    plan = SequentialPlan(context, pick_action)

    plan.parameter_infeerer.add_domains(
        EnumDomain(Arms), GraspDomain(GraspDescription, robot_view.left_arm.manipulator)
    )

    assert plan.parameter_infeerer.get_domain_for_type(Arms) == [
        Arms.LEFT,
        Arms.RIGHT,
        Arms.BOTH,
    ]

    grasp_domain = plan.parameter_infeerer.get_domain_for_type(GraspDescription)

    assert len(grasp_domain) == 12


def test_infer_domain_with_rules(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pick_action = PickUpActionDescription(
        world.get_body_by_name("milk.stl"),
        ...,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            robot_view.right_arm.manipulator,
        ),
    )

    plan = SequentialPlan(context, pick_action)

    plan.parameter_infeerer.add_domains(
        EnumDomain(Arms), GraspDomain(GraspDescription, robot_view.left_arm.manipulator)
    )

    plan.parameter_infeerer.add_rule(DummyRule(Arms, pick_action, "arm", 1))

    arm_domain = plan.parameter_infeerer.infer_domain_for_parameter(
        ParameterIdentifier(pick_action, "arm")
    )

    assert len(arm_domain) == 1
    assert arm_domain[0] == Arms.LEFT
