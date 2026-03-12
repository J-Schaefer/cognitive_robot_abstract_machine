import enum
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Iterable, TypeVar

from sqlalchemy.orm import sessionmaker

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.exceptions import (
    NoSolutionFound,
    GenerativeBackendQueryIsNotUnderspecifiedVariable,
)
from krrood.entity_query_language.factories import variable_from, set_of, variable
from krrood.entity_query_language.query.match import Match
from krrood.entity_query_language.query.query import Query
from krrood.ormatic.eql_interface import eql_to_sql
from krrood.parametrization.model_registries import (
    ModelRegistry,
    FullyFactorizedRegistry,
)
from krrood.parametrization.parameterizer import (
    MatchVariable,
    UnderspecifiedParameters,
)
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from probabilistic_model.probabilistic_model import ProbabilisticModel
from random_events.variable import Symbolic

T = TypeVar("T")


@dataclass
class QueryBackend(ABC):
    """
    Base class for all query backends.
    Query backends are objects that answer queries by different means.
    """

    @abstractmethod
    def evaluate(self, expression: Query) -> Iterable[T]:
        """
        Generate answers that match the expression.

        :param expression: The expression to generate answers for.
        :return: An iterable of answers.
        """


@dataclass
class SelectiveBackend(QueryBackend, ABC):
    """
    Selective backends are backends that select elements from existing data.
    These can take any query as input.
    """


@dataclass
class GenerativeBackend(QueryBackend, ABC):
    """
    Generative backends are backends that generate new elements.
    Generative backends have to take match expressions as input, since they need to construct new objects, and currently
    {py:class}`~krrood.entity_query_language.query.match.Match` is the only way to do so.
    """

    def evaluate(self, expression: Query) -> Iterable[T]:
        if not isinstance(expression, Match):
            raise GenerativeBackendQueryIsNotUnderspecifiedVariable(expression)
        yield from self._evaluate(expression)

    @abstractmethod
    def _evaluate(self, expression: Match[T]) -> Iterable[T]: ...


@dataclass
class SQLAlchemyBackend(SelectiveBackend):
    """
    A backend that selects elements from a database that is available via SQLAlchemy.
    """

    session_maker: sessionmaker
    """
    The session maker used for the database interactions.
    """

    def evaluate(self, expression: Query) -> Iterable:
        session = self.session_maker()
        translator = eql_to_sql(expression, session)
        yield from translator.evaluate()


@dataclass
class EntityQueryLanguageBackend(SelectiveBackend):
    """
    A domain that selects elements from a python process. This is just ordinary EQL.
    """

    def evaluate(self, expression: Query) -> Iterable:
        if isinstance(expression, Match):
            yield from self._evaluate_underspecified(expression)
        yield from expression.evaluate()

    def _evaluate_underspecified(self, expression: Match[T]) -> Iterable[T]:
        for attribute_match in expression.matches_with_variables:
            if isinstance(
                attribute_match.assigned_value, type(Ellipsis)
            ) and not issubclass(attribute_match.assigned_variable._type_, enum.Enum):
                raise ValueError(
                    f"Leaf statements in underspecified queries must be concrete objects or a symbolic expression."
                    f"If the assignment is Ellipsis it must, the type of the field must be an Enum, otherwise EQL cant"
                    f"generate it. If your looking for more flexible generations, try ProbabilisticBackend."
                    f"Got {attribute_match.name_from_variable_access_path} = {attribute_match.assigned_variable._type_}."
                )

            # convert ellipsis assignments for enum fields to symbolic expressions
            if isinstance(
                attribute_match.assigned_value, type(Ellipsis)
            ) and issubclass(attribute_match.assigned_variable._type_, enum.Enum):
                attribute_match.assigned_variable._value_ = variable(
                    attribute_match.assigned_variable._type_,
                    list(attribute_match.assigned_variable._type_),
                )

            # convert concrete objects to symbolic expressions
            else:
                attribute_match.assigned_variable._value_ = variable(
                    type(attribute_match.assigned_value),
                    [attribute_match.assigned_value],
                )
        all_combinations = set_of(
            *[
                attribute_match.assigned_variable
                for attribute_match in expression.matches_with_variables
            ]
        )
        for combination in all_combinations.evaluate():
            print(combination)


@dataclass
class ProbabilisticBackend(GenerativeBackend):
    """
    A backend that generates elements from a tractable probabilistic model using a model registry.
    """

    model_registry: ModelRegistry = field(default_factory=FullyFactorizedRegistry)
    """
    A model registry that can be used to resolve match statements to probabilistic models.
    """

    number_of_samples: int = field(kw_only=True, default=50)
    """
    The number of samples to generate.
    """

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:

        # generate parameters from example instance values
        parameters = UnderspecifiedParameters(expression)

        model = self.model_registry.get_model(parameters)

        # apply conditions from the parameters
        conditioned, _ = model.conditional(parameters.assignments_for_conditioning)

        if conditioned is None:
            raise NoSolutionFound(expression.expression)

        # apply conditions from the where statements
        if parameters.truncation_event:
            truncated, _ = conditioned.truncated(parameters.truncation_event)

            if truncated is None:
                raise NoSolutionFound(expression.expression)
        else:
            truncated = conditioned

        samples = truncated.sample(self.number_of_samples)

        # create new objects with the values from the samples
        for sample in samples:
            instance = parameters.create_instance_from_variables_and_sample(
                truncated.variables, sample
            )
            yield instance
