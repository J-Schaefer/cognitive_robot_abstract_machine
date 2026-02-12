from __future__ import annotations

import os.path
from abc import abstractmethod
import logging
from dataclasses import dataclass, fields

from typing_extensions import Any, Optional, Callable, TypeVar, Dict, Type

from krrood.entity_query_language.entity import variable, evaluate_condition
from krrood.entity_query_language.symbolic import Variable, SymbolicExpression
from ...designator import DesignatorDescription
from ...failures import PlanFailure, ConditionNotSatisfied

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ActionDescription(DesignatorDescription):
    _pre_perform_callbacks = []
    _post_perform_callbacks = []

    def __post_init__(self):
        pass
        # self._pre_perform_callbacks.append(self._update_robot_params)

    def perform(self) -> Any:
        """
        Full execution: pre-check, plan, post-check
        """
        logger.info(f"Performing action {self.__class__.__name__}")

        for pre_cb in self._pre_perform_callbacks:
            pre_cb(self)

        self.pre_condition()

        result = None
        try:
            result = self.execute()
        except PlanFailure as e:
            raise e
        finally:
            pass
            # for post_cb in self._post_perform_callbacks:
            #     post_cb(self)
            #
            # self.validate_postcondition(result)

        return result

    @abstractmethod
    def execute(self) -> Any:
        """
        Symbolic plan. Should only call motions or sub-actions.
        """
        pass

    @abstractmethod
    def pre_condition(self):
        pass

    @abstractmethod
    def post_condition(self):
        pass

    @property
    def validate_precondition(self) -> bool:
        """
        Symbolic/world state precondition validation.
        """
        return True

    @property
    def validate_postcondition(self) -> bool:
        """
        Symbolic/world state postcondition validation.
        """
        return True

    @classmethod
    def pre_perform(cls, func) -> Callable:
        cls._pre_perform_callbacks.append(func)
        return func

    @classmethod
    def post_perform(cls, func) -> Callable:
        cls._post_perform_callbacks.append(func)
        return func

    def get_bound_variables(self) -> Dict[T, Variable[T] | T]:
        self_fields = list(fields(self))
        [self_fields.remove(parent_field) for parent_field in fields(ActionDescription)]
        return {
            getattr(self, f.name): variable(
                type(getattr(self, f.name)), [getattr(self, f.name)]
            )
            for f in self_fields
        }

    def get_unbound_variables(self) -> Dict[T, Variable[T] | T]:
        self_fields = list(fields(self))
        [self_fields.remove(parent_field) for parent_field in fields(ActionDescription)]
        return {
            getattr(self, f.name): variable(
                type(getattr(self, f.name)),
                find_domain_for_type(type(getattr(self, f.name))),
            )
            for f in self_fields
        }

    def get_variables(self, unbound=False) -> Dict[T, Variable[T] | T]:
        # Maybe use python-box for a better interface
        return self.get_unbound_variables() if unbound else self.get_bound_variables()

    def evaluate_pre_condition(self) -> bool:
        evaluation = evaluate_condition(self.pre_condition())
        if evaluation:
            return True
        raise ConditionNotSatisfied(True, self.__class__, self.pre_condition())

    def evaluate_post_condition(self) -> bool:
        evaluation = evaluate_condition(self.post_condition())
        if evaluation:
            return True
        raise ConditionNotSatisfied(False, self.__class__, self.post_condition())


ActionType = TypeVar("ActionType", bound=ActionDescription)
