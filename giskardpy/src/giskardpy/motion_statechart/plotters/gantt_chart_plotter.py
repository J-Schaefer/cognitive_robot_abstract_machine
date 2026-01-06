from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from giskardpy.middleware import get_middleware
from giskardpy.motion_statechart.context import ExecutionContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.graph_node import Goal, MotionStatechartNode
from giskardpy.motion_statechart.plotters.styles import (
    LiftCycleStateToColor,
    ObservationStateToColor,
)
from giskardpy.utils.utils import create_path

if TYPE_CHECKING:
    from giskardpy.motion_statechart.motion_statechart import (
        MotionStatechart,
    )


@dataclass
class HistoryGanttChartPlotter:
    """
    Plot a hierarchy-aware Gantt chart of node states.

    Shows parent-child relationships of Goals by ordering rows in
    preorder and by prefixing labels with tree glyphs (├─, └─, │).
    Optional background bands and goal outlines emphasize grouping.
    """

    motion_statechart: MotionStatechart
    context: ExecutionContext | None = None
    second_width_in_cm: float = 2.0

    final_block_buffer: float = 1
    final_block_size: float = 2

    @property
    def x_width_per_control_cycle(self) -> float:
        if self.context is None:
            return 1
        return self.context.dt

    @property
    def total_control_cycles(self) -> int:
        return self.motion_statechart.history.history[-1].control_cycle

    @property
    def x_max(self) -> float:
        return (
            self.total_control_cycles + self.final_block_buffer + self.final_block_size
        )

    @property
    def num_bars(self) -> int:
        return len(self.motion_statechart.history.history[0].life_cycle_state)

    @property
    def use_seconds_for_x_axis(self) -> bool:
        return self.x_width_per_control_cycle != 1.0

    @property
    def figure_height(self) -> float:
        return 0.7 + self.num_bars * 0.25

    @property
    def figure_width(self) -> float:
        if not self.use_seconds_for_x_axis:
            return 0.5 * float((self.total_control_cycles or 0) + 1)
        # 1 inch = 2.54 cm; map seconds to figure width via second_length_in_cm
        inches_per_second = self.second_width_in_cm / 2.54
        return inches_per_second * self.time_span_seconds

    @property
    def time_span_seconds(self) -> float | None:
        return (
            self.total_control_cycles * self.x_width_per_control_cycle
            if self.x_width_per_control_cycle
            else None
        )

    def plot_gantt_chart(self, file_name: str) -> None:
        """
        Render the Gantt chart and save it.

        The chart shows life cycle (top half) and observation state (bottom half)
        per node over time. If a context with dt is provided, the x-axis is in seconds; otherwise, control cycles are used.
        """

        nodes = self.motion_statechart.nodes
        if len(nodes) == 0:
            get_middleware().logwarn(
                "Gantt chart skipped: no nodes in motion statechart."
            )
            return

        history = self.motion_statechart.history.history
        if len(history) == 0:
            get_middleware().logwarn("Gantt chart skipped: empty StateHistory.")
            return

        ordered_nodes = self._sort_nodes_by_parents()

        plt.figure(figsize=(self.figure_width, self.figure_height))
        plt.grid(True, axis="x", zorder=-1)

        for node_idx, node in enumerate(ordered_nodes):
            self._plot_lifecycle_bar(node=node, node_idx=node_idx)
            self._plot_observation_bar(node=node, node_idx=node_idx)

        self._format_axes(ordered_nodes=ordered_nodes)
        self._save_figure(file_name=file_name)

    def _sort_nodes_by_parents(self) -> List[MotionStatechartNode]:

        def return_children_in_order(n: MotionStatechartNode):
            yield n
            if isinstance(n, Goal):
                for c in n.nodes:
                    yield from return_children_in_order(c)

        ordered_: List[MotionStatechartNode] = []
        for root in self.motion_statechart.top_level_nodes:
            ordered_.extend(list(return_children_in_order(root)))
        # reverse list because plt plots bars bottom to top
        return list(reversed(ordered_))

    def _plot_lifecycle_bar(
        self,
        node: MotionStatechartNode,
        node_idx: int,
    ):
        life_cycle_history = (
            self.motion_statechart.history.get_life_cycle_history_of_node(node)
        )
        control_cycle_indices = [
            h.control_cycle for h in self.motion_statechart.history.history
        ]
        self._plot_node_bar(
            node_idx=node_idx,
            history=life_cycle_history,
            control_cycle_indices=control_cycle_indices,
            color_map=LiftCycleStateToColor,
            top=True,
        )

    def _plot_observation_bar(
        self,
        node: MotionStatechartNode,
        node_idx: int,
    ):
        obs_history = self.motion_statechart.history.get_observation_history_of_node(
            node
        )
        control_cycle_indices = [
            h.control_cycle for h in self.motion_statechart.history.history
        ]
        self._plot_node_bar(
            node_idx=node_idx,
            history=obs_history,
            control_cycle_indices=control_cycle_indices,
            color_map=ObservationStateToColor,
            top=False,
        )

    def _plot_node_bar(
        self,
        node_idx: int,
        history: List[LifeCycleValues | ObservationStateValues],
        control_cycle_indices: List[int],
        color_map: Dict[LifeCycleValues | ObservationStateValues, str],
        top: bool,
    ) -> None:
        current_state = history[0]
        start_idx = 0
        for idx, next_state in zip(control_cycle_indices[1:], history[1:]):
            if current_state != next_state:
                life_cycle_width = (idx - start_idx) * self.x_width_per_control_cycle
                self._draw_block(
                    node_idx=node_idx,
                    block_start=start_idx * self.x_width_per_control_cycle,
                    block_width=life_cycle_width,
                    color=color_map[current_state],
                    top=top,
                )
                start_idx = idx
                current_state = next_state
        # plot last tick
        last_idx = control_cycle_indices[-1]
        life_cycle_width = (last_idx - start_idx) * self.x_width_per_control_cycle
        self._draw_block(
            node_idx=node_idx,
            block_start=start_idx * self.x_width_per_control_cycle,
            block_width=life_cycle_width,
            color=color_map[current_state],
            top=top,
        )
        block_start = start_idx * self.x_width_per_control_cycle + life_cycle_width

        # plot white buffer block
        self._draw_block(
            node_idx=node_idx,
            block_start=block_start,
            block_width=self.final_block_buffer,
            color="white",
            top=top,
        )
        block_start += self.final_block_buffer

        # plot last tick
        self._draw_block(
            node_idx=node_idx,
            block_start=block_start,
            block_width=self.final_block_size,
            color=color_map[current_state],
            top=top,
        )

    def _draw_block(
        self,
        node_idx,
        block_start,
        block_width,
        color,
        top: bool,
        bar_height: float = 0.8,
    ):
        if top:
            y = node_idx + bar_height / 4
        else:
            y = node_idx - bar_height / 4
        plt.barh(
            y,
            block_width,
            height=bar_height / 2,
            left=block_start,
            color=color,
            zorder=2,
        )

    def _format_axes(
        self,
        ordered_nodes: List[MotionStatechartNode],
    ) -> None:
        if self.use_seconds_for_x_axis:
            #     total_seconds = self.x_max * self.x_width_per_control_cycle
            plt.xlabel("Time [s]")
            #     plt.xlim(0, total_seconds)
            base_ticks = np.arange(0.0, self.time_span_seconds + 1e-9, 0.5).tolist()
        #     plt.xticks(ticks)
        else:
            plt.xlabel("Control cycle")
            step = self.x_width_per_control_cycle
            base_ticks = list(range(0, self.total_control_cycles + 1, step))

        blank_pos = self.time_span_seconds + self.final_block_buffer
        final_pos = blank_pos + self.final_block_size // 2
        final_blank_pose = final_pos + self.final_block_size // 2

        plt.xlim(0, final_blank_pose)

        tick_positions = base_ticks + [final_pos]
        tick_labels = [str(t) for t in base_ticks] + ["final"]

        plt.xticks(tick_positions, tick_labels)

        plt.ylabel("Nodes")
        num_bars = len(self.motion_statechart.history.history[0].life_cycle_state)
        plt.ylim(-0.8, num_bars - 1 + 0.8)

        node_names = []
        for idx, n in enumerate(ordered_nodes):
            if idx == 0:
                prev_depth = 0
            else:
                prev_depth = ordered_nodes[idx - 1].depth
            node_names.append(self._make_label(n, prev_depth))
        node_idx = list(range(len(node_names)))
        plt.yticks(node_idx, node_names)
        plt.gca().yaxis.tick_right()
        plt.tight_layout()

    def _make_label(self, node: MotionStatechartNode, prev_depth: int) -> str:
        depth = node.depth
        if depth == 0:
            return node.unique_name
        diff = depth - prev_depth
        if diff > 0:
            return (
                "│  " * (depth - diff)
                + "└─"  # no space because the formatting is weird otherwise
                * (diff - 1)
                + "└─ "
                + node.unique_name
            )
        else:
            return "│  " * (depth - 1) + "├─ " + node.unique_name

    def _save_figure(self, file_name: str) -> None:
        create_path(file_name)
        plt.savefig(file_name)
        plt.close()
        get_middleware().loginfo(f"Saved gantt chart to {file_name}.")
