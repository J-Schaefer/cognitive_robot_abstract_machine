# used for delayed evaluation of typing until python 3.11 becomes mainstream
from __future__ import annotations

import atexit
import logging
import threading
from dataclasses import dataclass, field
from queue import Queue

from typing_extensions import (
    Optional,
    Callable,
    Any,
    List,
    Union,
    Self,
    TYPE_CHECKING,
)

from pycram.datastructures.enums import TaskStatus, MonitorBehavior
from pycram.failures import PlanFailure
from pycram.fluent import Fluent

from pycram.plans.plan_node import (
    PlanNode,
)

if TYPE_CHECKING:
    from pycram.plans.plan import (
        Plan,
    )

from pycram.ros import sleep


logger = logging.getLogger(__name__)


@dataclass(eq=False)
class LanguageNode(PlanNode):
    """
    Base class for language nodes in a plan.
    Language nodes are nodes that are not directly executable, but manage the execution of their children in a certain
    way.
    """

    @classmethod
    def with_children(cls, children: List[PlanNode], plan: Plan) -> Self:
        """
        A factory to create a LanguageNode with the given children.

        :param children: The children to be added to the LanguageNode
        :return: The created LanguageNode
        """
        result = cls()
        plan.add_node(result)
        for child in children:
            plan.add_edge(result, child)
        return result


@dataclass
class SequentialNode(LanguageNode):
    """
    Executes all children sequentially, an exception while executing a child does not terminate the whole process.
    Instead, the exception is saved to a list of all exceptions thrown during execution and returned.

    Behaviour:
        Returns a tuple containing the final state of execution (SUCCEEDED, FAILED) and a list of results from each
        child's perform() method. The state is :py:attr:`~TaskStatus.SUCCEEDED` *iff* all children are executed without
        exception. In any other case the State :py:attr:`~TaskStatus.FAILED` will be returned.
    """

    def perform(self):
        """
        Behaviour of Sequential, calls perform() on each child sequentially

        :return: The state and list of results according to the behaviour described in :func:`Sequential`
        """
        result = None
        try:
            logger.info(f"Executing {self}")
            result = self.perform_sequential(self.children)
            self.status = TaskStatus.SUCCEEDED
        except PlanFailure as e:
            self.status = TaskStatus.FAILED
            self.reason = e
            # Failure Handling could be done here
            raise e
        return result

    def perform_sequential(self, nodes: List[PlanNode], raise_exceptions=True) -> Any:
        """
        Behavior of the sequential node, performs all children in sequence and raises error if they occur.

        :param nodes: A list of nodes which should be performed in sequence
        :param raise_exceptions: If True (default) errors will be raised
        """
        res = None
        for child in nodes:
            try:
                res = child.perform()
            except PlanFailure as e:
                self.status = TaskStatus.FAILED
                self.reason = e
                if raise_exceptions:
                    raise e

        return res


@dataclass
class ParallelNode(LanguageNode):
    """
    Executes all children in parallel by creating a thread per children and executing them in the respective thread. All
    exceptions during execution will be caught, saved to a list and returned upon end.

    Behaviour:
        Returns a tuple containing the final state of execution (SUCCEEDED, FAILED) and a list of results from
        each child's perform() method. The state is :py:attr:`~TaskStatus.SUCCEEDED` *iff* all children could be executed without
        an exception. In any other case the State :py:attr:`~TaskStatus.FAILED` will be returned.

    """

    def perform(self):
        """
        Behaviour of Parallel, creates a new thread for each child and calls perform() of the child in the respective
        thread.

        :return: The state and list of results according to the behaviour described in :func:`Parallel`

        """
        self.perform_parallel(self.children)
        child_statuses = [child.status for child in self.children]
        self.status = (
            TaskStatus.SUCCEEDED
            if TaskStatus.FAILED not in child_statuses
            else TaskStatus.FAILED
        )

    def perform_parallel(self, nodes: List[PlanNode]):
        """
        Behaviour of the parallel node performs the given nodes in parallel in different threads.

        :param nodes: A list of nodes which should be performed in parallel
        """
        threads = []
        for child in nodes:
            t = threading.Thread(target=self._lang_call, args=[child])
            t.start()
            threads.append(t)
        for thread in threads:
            thread.join()

    def _lang_call(self, node: PlanNode):
        """
        Wrapper method which is executed in the thread. Wraps the given node in a try catch and performs it

        :param node: The node which is to be performed
        """
        try:
            return node.perform()
        except PlanFailure as e:
            self.status = TaskStatus.FAILED
            self.reason = e
            # Failure handling comes here
            raise e


@dataclass
class RepeatNode(SequentialNode):
    """
    Executes all children a given number of times in sequential order.
    """

    repetitions: int = 1
    """
    The number of repetitions of the children.
    """

    def perform(self):
        """
        Behaviour of repeat, executes all children in a loop as often as stated on initialization.

        :return:
        """
        for _ in range(self.repetitions):
            try:
                self.perform_sequential(self.children)
            except PlanFailure as e:
                self.status = TaskStatus.FAILED
                self.reason = e
                raise e

    def __hash__(self):
        return id(self)


@dataclass
class MonitorNode(SequentialNode):
    """
    Monitors a Language Expression and interrupts it when the given condition is evaluated to True.

    Behaviour:
        Monitors start a new Thread which checks the condition while performing the nodes below it. Monitors can have
        different behaviors, they can Interrupt, Pause or Resume the execution of the children.
        If the behavior is set to Resume the plan will be paused until the condition is met.
    """

    def __init__(
        self,
        condition: Union[Callable, Fluent] = None,
        behavior: Optional[MonitorBehavior] = MonitorBehavior.INTERRUPT,
    ):
        """
        When initializing a Monitor a condition must be provided. The condition is a callable or a Fluent which returns \
        True or False.

        :param condition: The condition upon which the Monitor should interrupt the attached language expression.
        """
        super().__init__()
        self.kill_event = threading.Event()
        self.exception_queue = Queue()
        self.behavior = behavior
        if self.behavior == MonitorBehavior.RESUME:
            self.pause()
        if callable(condition):
            self.condition = Fluent(condition)
        elif isinstance(condition, Fluent):
            self.condition = condition
        else:
            raise AttributeError(
                "The condition of a Monitor has to be a Callable or a Fluent"
            )
        self.monitor_thread = threading.Thread(
            target=self.monitor, name=f"MonitorThread-{id(self)}"
        )
        self.monitor_thread.start()

    def perform(self):
        """
        Behavior of the Monitor, starts a new Thread which checks the condition and then performs the attached language
        expression

        :return: The state of the attached language expression, as well as a list of the results of the children
        """
        self.perform_sequential(self.children)
        self.kill_event.set()
        self.monitor_thread.join()

    def monitor(self):
        atexit.register(self.kill_event.set)
        while not self.kill_event.is_set():
            if self.condition.get_value():
                if self.behavior == MonitorBehavior.INTERRUPT:
                    self.interrupt()
                    self.kill_event.set()
                elif self.behavior == MonitorBehavior.PAUSE:
                    self.pause()
                    self.kill_event.set()
                elif self.behavior == MonitorBehavior.RESUME:
                    self.resume()
                    self.kill_event.set()
            sleep(0.1)

    def __hash__(self):
        return id(self)


@dataclass
class TryInOrderNode(SequentialNode):
    """
    Executes all children sequentially, an exception while executing a child does not terminate the whole process.
    Instead, the exception is saved to a list of all exceptions thrown during execution and returned.

    Behaviour:
        Returns a tuple containing the final state of execution (SUCCEEDED, FAILED) and a list of results from each
        child's perform() method. The state is :py:attr:`~TaskStatus.SUCCEEDED` if one or more children are executed without
        exception. In the case that all children could not be executed the State :py:attr:`~TaskStatus.FAILED` will be returned.
    """

    def perform(self):
        """
        Behaviour of TryInOrder, calls perform() on each child sequentially and catches raised exceptions.

        :return: The state and list of results according to the behaviour described in :func:`TryInOrder`
        """
        self.perform_sequential(self.children, raise_exceptions=False)
        child_statuses = [child.status for child in self.children]
        self.status = (
            TaskStatus.SUCCEEDED
            if TaskStatus.SUCCEEDED in child_statuses
            else TaskStatus.FAILED
        )
        child_results = list(
            filter(None, [child.result for child in self.recursive_children])
        )
        if child_results:
            self.result = child_results[0]
            return self.result[0]

    def __hash__(self):
        return id(self)


@dataclass
class TryAllNode(ParallelNode):
    """
    Executes all children in parallel by creating a thread per children and executing them in the respective thread. All
    exceptions during execution will be caught, saved to a list and returned upon end.

    Behaviour:
        Returns a tuple containing the final state of execution (SUCCEEDED, FAILED) and a list of results from each
        child's perform() method. The state is :py:attr:`~TaskStatus.SUCCEEDED` if one or more children could be executed
        without raising an exception. If all children fail the State :py:attr:`~TaskStatus.FAILED` will be returned.
    """

    def perform(self):
        """
        Behaviour of TryAll, creates a new thread for each child and executes all children in their respective threads.

        :return: The state and list of results according to the behaviour described in :func:`TryAll`
        """
        self.perform_parallel(self.children)
        child_statuses = [child.status for child in self.children]
        self.status = (
            TaskStatus.SUCCEEDED
            if TaskStatus.SUCCEEDED in child_statuses
            else TaskStatus.FAILED
        )

    def __hash__(self):
        return id(self)


@dataclass
class CodeNode(LanguageNode):
    """
    Executable code block in a plan.
    """

    code: Callable = field(default_factory=lambda: lambda: None, kw_only=True)

    def execute(self) -> Any:
        """
        Execute the code with its arguments

        :returns: Anything that the function associated with this object will return.
        """
        return self.code()
