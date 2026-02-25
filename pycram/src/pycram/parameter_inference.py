from __future__ import annotations

import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from enum import Enum
from itertools import product

from sortedcontainers import SortedSet

from typing_extensions import (
    Any,
    List,
    TypeVar,
    Generic,
    Type,
    Dict,
    TYPE_CHECKING,
    Tuple,
)

from pycram.datastructures.enums import ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import PoseStamped
from pycram.utils import get_all_values_in_enum
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    KinematicStructureEntity,
)
from .datastructures.dataclasses import Context

if TYPE_CHECKING:
    from .plan import Plan, ActionDescriptionNode
    from .datastructures.partial_designator import PartialDesignator

logger = logging.getLogger("pycram")

T = TypeVar("T")


def find_domain_for_value(value: Any, world: World) -> List:
    """
    Given a value finds the possible domain of values for in the world. A domain of values is a list of all values
    that could be used.

    :param value: The value to find a domain for
    :param world: The world in which should be searched
    :return: A list of possible values
    """
    value_type = type(value)
    if issubclass(value_type, SemanticAnnotation):
        return [
            sa
            for sa in world.semantic_annotations
            if issubclass(type(sa), (value_type, SemanticAnnotation))
        ]
    elif issubclass(value_type, KinematicStructureEntity):
        # return world.kinematic_structure_entities
        return [value]
    elif issubclass(value_type, Enum):
        return get_all_values_in_enum(value_type)
    elif issubclass(value_type, PoseStamped):
        return [value]
    elif issubclass(value_type, GraspDescription):
        return [
            GraspDescription(approach, align, value.manipulator)
            for approach, align in product(
                get_all_values_in_enum(ApproachDirection),
                get_all_values_in_enum(VerticalAlignment),
            )
        ]
    logger.warning(f"There is no domain for type {value_type}")
    return []


def find_domain_for_type(value_type, world: World):
    pass


@dataclass
class ParameterInferrer:
    """
    Central module to manage the infeerence of domains and parameter.

    Principle:
        Domains define general space of values for a type and are defined per type
        Rules: restrict the domain and are defined for certain parameters
    """

    parameter_rules: List[ParameterInferenceRule] = field(
        init=False, default_factory=list
    )
    """
    A set of rules that restrict the domain
    """

    type_domains: List[Domain] = field(init=False, default_factory=list)
    """
    Domains for all defined types
    """

    plan: Plan = None
    """
    Back-reference to the plan to which this infeerer belongs 
    """

    def add_rule(self, inference_rule: InferenceRule):
        """
        Adds a rule to the set of restriction rules

        :param inference_rule: The rule to add
        """
        self.parameter_rules.append(inference_rule)

    def add_domain(self, domain: Domain[Type[T]]):
        """
        Adds a domain to the set of domains

        :param domain: The domain to add
        """
        self.type_domains.append(domain)

    def add_domains(self, *domains: Domain[Type[T]]):
        """
        Adds a set set of domains
        """
        for domain in domains:
            self.add_domain(domain)

    def get_domain_for_type(self, type_):
        """
        Finds the domain specification for a given type. The domain is the union of all Domains that are specified for
        the given type.

        :param type_: Type for which to find the domain
        :return: The domain
        """
        result = set()
        for domain in self.type_domains:
            if domain.domain_type == type_:
                result.update(domain.domain(self.plan.context))
        return list(result)

    def get_rules_for_parameter(self, parameter_identifier: ParameterIdentifier):
        """
        Finds all rules that are applicable for a parameter

        :param parameter_identifier: Identification for the parameter
        :return: A list of rules
        """
        return [
            rule
            for rule in self.parameter_rules
            if rule.action_description == parameter_identifier.action_description
            and rule.parameter_name == parameter_identifier.parameter_name
        ]

    def infer_domain_for_parameter(self, parameter_identifier: ParameterIdentifier):
        """
        Finds the domain for the type and then applies rules and their effects.

        :param parameter_identifier: Identification for the parameter
        :return: Domain for the parameter
        """
        domain = self.get_domain_for_type(parameter_identifier.type_)
        rules = self.get_rules_for_parameter(parameter_identifier)
        for rule in rules:
            domain = rule.apply(domain, self.plan.context)
            rule.effect()
        return domain

    def sample_value(self, value_type: Type[T]) -> T:
        pass


@dataclass
class InferenceRule(Generic[T], ABC):
    """
    Rule that restricts a domain
    """

    parameter_type: Type[T]
    """
    Type for which this rule is defined (??)
    """

    @abstractmethod
    def apply(self, domain: List[T], context: Context) -> List[T]: ...

    """
    Applies this rule to the domain
    """

    def effect(self): ...

    """
    Effect that is triggered ones this rule fires. 
    """


@dataclass
class ParameterIdentifier:
    """
    Identifies a specific parameter of an action description
    """

    action_description: PartialDesignator
    """
    Description of an action to which the parameter belongs
    """

    parameter_name: str
    """
    Name of the parameter
    """

    @property
    def parameter(self) -> T:
        return self.action_description.kwargs[self.parameter_name]

    @property
    def type_(self) -> Type[T]:
        action_types = {
            f.name: f.type for f in self.action_description.performable.fields
        }
        return action_types[self.parameter_name]

    @property
    def path(self) -> str:
        return f"{self.action_description.performable}.{self.parameter_name}"

    def __hash__(self):
        return id(self)


@dataclass
class Domain(Generic[T], ABC):
    """
    Defines a domain of values. A domain are all possible values that a type can be.
    """

    domain_type: Type[T]
    """
    The type over which the domain is defined
    """

    @abstractmethod
    def domain(self, context: Context) -> List[T]: ...


@dataclass
class ParameterInferenceRule(ParameterIdentifier, InferenceRule, ABC): ...


@dataclass
class TypeInferenceRule(InferenceRule, ABC): ...
