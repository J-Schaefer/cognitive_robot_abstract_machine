import unittest
from unittest.mock import MagicMock
import sys
import os

# Create a fake Qt Application for testing widgets
from PySide6.QtWidgets import QApplication

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

from probabilistic_model.gui.controller import ModelController
from probabilistic_model.gui.query_widget import QueryWidget
from probabilistic_model.gui.variable_constraint_widget import VariableConstraintWidget
from random_events.variable import Continuous, Symbolic
from random_events.set import Set, SetElement
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent


class TestQueryGUI(unittest.TestCase):

    def setUp(self):
        self.controller = ModelController()

        # Setup a simple model
        self.v1 = Continuous("v1", closed(0, 1))
        self.v2 = Symbolic("v2", Set.from_iterable(["a", "b"]))

        self.model = MagicMock()
        self.model.variables = [self.v1, self.v2]
        self.model.probability.return_value = 0.5

        self.controller.set_model(self.model)

    def test_variable_constraint_widget_continuous(self):
        widget = VariableConstraintWidget(self.model.variables)
        changed_mock = MagicMock()
        widget.changed.connect(changed_mock)

        # Select v1
        index = widget.variable_combo.findText("v1")
        widget.variable_combo.setCurrentIndex(index)

        self.assertEqual(widget.variable_combo.currentData(), self.v1)
        self.assertIsNotNone(widget.constraint_widget)
        self.assertTrue(changed_mock.called)
        changed_mock.reset_mock()

        # Check slider value change triggers changed signal
        slider = widget.constraint_widget
        slider.setValue((100, 900))
        self.assertTrue(changed_mock.called)

        # Check constraint retrieval
        var, constraint = widget.get_constraint()
        self.assertEqual(var, self.v1)
        self.assertTrue(isinstance(constraint, type(closed(0, 1))))

    def test_variable_constraint_widget_symbolic(self):
        widget = VariableConstraintWidget(self.model.variables)
        # Select v2
        index = widget.variable_combo.findText("v2")
        widget.variable_combo.setCurrentIndex(index)

        self.assertEqual(widget.variable_combo.currentData(), self.v2)
        self.assertIsNotNone(widget.constraint_widget)

        # Check constraint retrieval
        var, constraint = widget.get_constraint()
        self.assertEqual(var, self.v2)
        self.assertTrue(isinstance(constraint, Set))

    def test_query_widget_instantiation(self):
        widget = QueryWidget(self.controller)
        self.assertIsNotNone(widget)
        self.assertEqual(len(widget.query_widgets), 1)
        self.assertEqual(len(widget.evidence_widgets), 1)

    def test_query_widget_add_remove_row(self):
        widget = QueryWidget(self.controller)
        initial_count = len(widget.query_widgets)

        # Trigger add row for query
        # Since add_variable_row is connected to button, we can call it directly
        # or simulate button click if we find the button.
        # For simplicity, call the method.
        layout_container = widget.query_widgets[0].parentWidget().parentWidget()
        widget.add_variable_row(widget.query_widgets, layout_container)

        self.assertEqual(len(widget.query_widgets), initial_count + 1)

        # Remove row
        row_widget = widget.query_widgets[-1].parentWidget()
        var_widget = widget.query_widgets[-1]
        widget.remove_variable_row(row_widget, var_widget, widget.query_widgets)
        self.assertEqual(len(widget.query_widgets), initial_count)

    def test_controller_calculate_probability(self):
        query = Event(SimpleEvent({self.v1: closed(0, 0.5)}))
        evidence = Event(SimpleEvent({self.v2: Set.from_iterable(["a"])}))

        self.model.probability.return_value = 0.5
        # Mock intersection_with
        joint = query.intersection_with(evidence)

        prob = self.controller.calculate_probability(query, evidence)
        self.model.probability.assert_any_call(evidence)
        self.assertEqual(prob, 1.0)  # 0.5 / 0.5 in this mock case

    def test_variable_constraint_widget_labels(self):
        # We need priors to test marks from priors
        from random_events.product_algebra import VariableMap

        priors = VariableMap()
        # Mocking a distribution that has support
        dist = MagicMock()
        # support is usually a SimpleEvent in simple cases, but in CompositeSets it's multiple
        # Actually in random_events, distribution.support is an Event (CompositeSet)
        dist.support.simple_sets = [SimpleEvent({self.v1: closed(0.1, 0.9)})]
        priors[self.v1] = dist

        widget = VariableConstraintWidget(self.model.variables, priors)
        # Select v1
        index = widget.variable_combo.findText("v1")
        widget.variable_combo.setCurrentIndex(index)

        # Check if value_label exists and has text
        self.assertTrue(hasattr(widget, "value_label"))
        self.assertIn("Range:", widget.value_label.text())
        self.assertIn("0.10", widget.value_label.text())
        self.assertIn("0.90", widget.value_label.text())

        # Check for marks labels
        from PySide6.QtWidgets import QLabel

        labels = widget.findChildren(QLabel)
        mark_labels = [l for l in labels if l != widget.value_label]
        # Should have around 6 marks
        self.assertGreaterEqual(len(mark_labels), 2)
        # Verify one of the marks is the minimum
        mark_texts = [l.text() for l in mark_labels]
        self.assertIn("0.10", mark_texts)
        self.assertIn("0.90", mark_texts)


if __name__ == "__main__":
    unittest.main()
