from dataclasses import dataclass
from itertools import product

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import ApproachDirection, VerticalAlignment, Arms
from pycram.datastructures.grasp import GraspDescription
from pycram.parameter_inference import Domain
from pycram.plan import Plan
from pycram.utils import get_all_values_in_enum
from semantic_digital_twin.robots.abstract_robot import Manipulator


@dataclass
class EnumDomain(Domain):

    def domain(self, context: Context):
        return get_all_values_in_enum(self.domain_type)


@dataclass
class GraspDomain(Domain):

    manipulator: Manipulator

    def domain(self, context: Context):
        return [
            GraspDescription(approach, align, self.manipulator)
            for approach, align in product(
                get_all_values_in_enum(ApproachDirection),
                get_all_values_in_enum(VerticalAlignment),
            )
        ]


@dataclass
class SemanticAnnotationDomain(Domain):

    def domain(self, context: Context):
        return context.world.get_semantic_annotations_by_type(self.domain_type)


def load_default_domains(plan: Plan):
    plan.parameter_infeerer.add_domains(EnumDomain(Arms))
