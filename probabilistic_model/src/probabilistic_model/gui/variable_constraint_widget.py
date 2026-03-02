from typing import Optional, List, Union
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QComboBox,
    QListWidget,
    QAbstractItemView,
    QVBoxLayout,
    QLabel,
)
from PySide6.QtCore import Signal, Qt
from superqt import QRangeSlider

from random_events.variable import Variable, Continuous, Symbolic, Integer
from random_events.product_algebra import SimpleEvent, Event, VariableMap
from random_events.interval import closed, Interval
from random_events.set import Set, SetElement


class VariableConstraintWidget(QWidget):
    """
    A widget that allows selecting a variable and defining its constraints.
    """

    changed = Signal()

    def __init__(
        self,
        variables: List[Variable],
        priors: Optional[VariableMap] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.variables = sorted(variables, key=lambda v: v.name)
        self.priors = priors
        self.init_ui()

    def init_ui(self):
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Variable selector
        self.variable_combo = QComboBox()
        self.variable_combo.addItem("Select Variable...", None)
        for var in self.variables:
            self.variable_combo.addItem(var.name, var)

        self.variable_combo.currentIndexChanged.connect(self.on_variable_changed)
        self.layout.addWidget(self.variable_combo, 1)

        # Container for constraint input
        self.constraint_container = QWidget()
        self.constraint_layout = QVBoxLayout(self.constraint_container)
        self.constraint_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.constraint_container, 2)

        self.constraint_widget = None

    def on_variable_changed(self):
        # Clear previous constraint widgets
        while self.constraint_layout.count():
            item = self.constraint_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            else:
                # Could be a layout
                sub_layout = item.layout()
                if sub_layout:
                    # Clear sub_layout
                    while sub_layout.count():
                        sub_item = sub_layout.takeAt(0)
                        sub_widget = sub_item.widget()
                        if sub_widget:
                            sub_widget.deleteLater()
        self.constraint_widget = None

        variable = self.variable_combo.currentData()
        if variable:
            self.create_constraint_widget(variable)

        self.changed.emit()

    def create_constraint_widget(self, variable: Variable):
        if variable.is_numeric:
            # Range Slider for Continuous or Integer
            # Default to domain
            try:
                mini = variable.domain.simple_sets[0].lower
                maxi = variable.domain.simple_sets[-1].upper
            except (AttributeError, IndexError):
                # Fallback for composite sets
                try:
                    mini = variable.domain.simple_sets[0].simple_sets[0].lower
                    maxi = variable.domain.simple_sets[-1].simple_sets[-1].upper
                except (AttributeError, IndexError):
                    mini, maxi = 0, 1

            # Try to get from priors (support)
            if self.priors and variable in self.priors:
                try:
                    support = self.priors[variable].support
                    if support.simple_sets:
                        mini = support.simple_sets[0][variable].simple_sets[0].lower
                        maxi = support.simple_sets[-1][variable].simple_sets[-1].upper
                except Exception:
                    pass

            # Handle infinity
            if mini == float("-inf"):
                mini = -100.0
            if maxi == float("inf"):
                maxi = 100.0

            # Handle equality
            if mini == maxi:
                mini -= 1.0
                maxi += 1.0

            # Current value label
            self.value_label = QLabel()
            self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.constraint_layout.addWidget(self.value_label)

            slider = QRangeSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(
                int(mini * 1000) if not isinstance(variable, Integer) else int(mini)
            )
            slider.setMaximum(
                int(maxi * 1000) if not isinstance(variable, Integer) else int(maxi)
            )
            slider.setValue((slider.minimum(), slider.maximum()))

            def update_label(val):
                low, high = val
                if not isinstance(variable, Integer):
                    low /= 1000.0
                    high /= 1000.0
                self.value_label.setText(f"Range: [{low:.2f}, {high:.2f}]")

            slider.valueChanged.connect(update_label)
            update_label(slider.value())

            slider.valueChanged.connect(lambda _: self.changed.emit())
            self.constraint_widget = slider
            self.constraint_layout.addWidget(self.constraint_widget)

            # Marks
            marks_layout = QHBoxLayout()
            size = maxi - mini
            if size > 0:
                import numpy as np

                steps = [x for x in np.arange(mini, maxi, size / 5)] + [maxi]
                for step in steps:
                    text = (
                        f"{int(round(step))}"
                        if isinstance(variable, Integer)
                        else f"{step:.2f}"
                    )
                    mark_label = QLabel(text)
                    mark_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    mark_label.setStyleSheet("font-size: 8pt; color: gray;")
                    marks_layout.addWidget(mark_label)
            self.constraint_layout.addLayout(marks_layout)

        else:
            # List Widget for Symbolic (multi-selection)
            list_widget = QListWidget()
            list_widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)

            # variable.domain is a Set for Symbolic variables
            for element in variable.domain.all_elements:
                list_widget.addItem(str(element))
                item = list_widget.item(list_widget.count() - 1)
                item.setSelected(True)  # Default select all

            list_widget.itemSelectionChanged.connect(self.changed.emit)
            # Set a reasonable height for the list
            list_widget.setMaximumHeight(100)
            self.constraint_widget = list_widget

        self.constraint_layout.addWidget(self.constraint_widget)

    def get_constraint(self) -> Optional[tuple[Variable, Union[Interval, Set]]]:
        variable = self.variable_combo.currentData()
        if not variable or not self.constraint_widget:
            return None

        if isinstance(variable, (Continuous, Integer)):
            slider: QRangeSlider = self.constraint_widget
            low, high = slider.value()
            if not isinstance(variable, Integer):
                low /= 1000.0
                high /= 1000.0
            return variable, closed(low, high)
        elif isinstance(variable, Symbolic):
            list_widget: QListWidget = self.constraint_widget
            selected_items = list_widget.selectedItems()
            selected_values = [item.text() for item in selected_items]
            # Create a Set from selected values
            # Need to be careful with types here, selected_values are strings
            # and the original domain might have different types.
            # For now, assuming strings or using match.
            all_elements = variable.domain.all_elements
            matched_elements = [e for e in all_elements if str(e) in selected_values]

            if not matched_elements:
                return variable, Set()  # Empty set

            return variable, Set(
                *[SetElement(e, all_elements) for e in matched_elements]
            )

        return None

    def set_constraint(self, variable: Variable, constraint: Union[Interval, Set]):
        """
        Sets the variable and its constraint programmatically.
        """
        # Select the variable in the combo box
        for i in range(self.variable_combo.count()):
            data = self.variable_combo.itemData(i)
            if data is not None and data == variable:
                self.variable_combo.setCurrentIndex(i)
                break

        # Now the constraint widget should be created via on_variable_changed
        if not self.constraint_widget:
            return

        if isinstance(variable, (Continuous, Integer)):
            slider: QRangeSlider = self.constraint_widget
            # constraint is an Interval
            if not constraint.simple_sets:
                return
            first_interval = constraint.simple_sets[0]
            low = first_interval.lower
            high = first_interval.upper
            if not isinstance(variable, Integer):
                low *= 1000.0
                high *= 1000.0
            slider.setValue((int(low), int(high)))
        elif isinstance(variable, Symbolic):
            list_widget: QListWidget = self.constraint_widget
            # constraint is a Set
            selected_str_values = [str(e.element) for e in constraint.simple_sets]
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                item.setSelected(item.text() in selected_str_values)
