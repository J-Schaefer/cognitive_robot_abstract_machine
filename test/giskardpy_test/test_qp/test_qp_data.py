import numpy as np
import pytest
import scipy.sparse as sp
from giskardpy.qp.qp_data import QPData, Conditioning
from giskardpy.qp.solvers.qp_solver_qpSWIFT import QPSolverQPSwift


@pytest.fixture(scope="module")
def simple_inequality_qp() -> tuple[QPData, np.ndarray]:
    """
    min_{x1, x2} x1^2 + x2^2
    s.t.
        -inf <= x1 <= inf
        -inf <= x2 <= inf
        -inf <= x1 + x2 <= -0.5

    expected solution:
        [-0.25, -0.25]
    """
    return QPData(
        quadratic_weights=np.array([1.0, 1.0]),
        linear_weights=np.array([0.0, 0.0]),
        box_lower_constraints=np.array([-np.inf, -np.inf]),
        box_upper_constraints=np.array([np.inf, np.inf]),
        eq_matrix=sp.csc_matrix(np.zeros((0, 2))),
        eq_bounds=np.array([]),
        neq_matrix=sp.csc_matrix(np.array([[1.0, 1.0]])),
        neq_lower_bounds=np.array([-np.inf]),
        neq_upper_bounds=np.array([-0.5]),
    ), np.array([-0.25, -0.25])


@pytest.fixture(scope="module")
def simple_eq_as_inequality_qp() -> tuple[QPData, np.ndarray]:
    """
    min_{x1, x2} x1^2 + x2^2
    s.t.
        -inf <= x1 <= inf
        -inf <= x2 <= inf
        -inf <= x1 + x2 <= -0.5

    expected solution:
        [-0.25, -0.25]
    """
    return QPData(
        quadratic_weights=np.array([1.0, 1.0]),
        linear_weights=np.array([0.0, 0.0]),
        box_lower_constraints=np.array([-np.inf, -np.inf]),
        box_upper_constraints=np.array([np.inf, np.inf]),
        eq_matrix=sp.csc_matrix(np.zeros((0, 2))),
        eq_bounds=np.array([]),
        neq_matrix=sp.csc_matrix(np.array([[1.0, 1.0]])),
        neq_lower_bounds=np.array([-0.5]),
        neq_upper_bounds=np.array([-0.5]),
    ), np.array([-0.25, -0.25])


@pytest.fixture(scope="module")
def simple_equality_qp() -> tuple[QPData, np.ndarray]:
    """
    min_{x1, x2} x1^2 + x2^2
    s.t.
        -inf <= x1 <= inf
        -inf <= x2 <= inf
        x1 + x2 = -0.5

    expected solution:
        [-0.25, -0.25]
    """
    return QPData(
        quadratic_weights=np.array([1.0, 1.0]),
        linear_weights=np.array([0.0, 0.0]),
        box_lower_constraints=np.array([-np.inf, -np.inf]),
        box_upper_constraints=np.array([np.inf, np.inf]),
        eq_matrix=sp.csc_matrix(np.array([[1.0, 1.0]])),
        eq_bounds=np.array([-0.5]),
        neq_matrix=sp.csc_matrix(np.zeros((0, 2))),
        neq_lower_bounds=np.array([]),
        neq_upper_bounds=np.array([]),
    ), np.array([-0.25, -0.25])


@pytest.fixture(scope="module")
def box_constraints_qp() -> tuple[QPData, np.ndarray]:
    """
    min_{x1, x2} 0.5 * (2*x1^2 + 2*x2^2)
    s.t.
        -1.0 <= x1 <= 1.0
        0.5 <= x2 <= 2.0

    expected solution:
        [0.0, 0.5]
    """
    return QPData(
        quadratic_weights=np.array([2.0, 2.0]),
        linear_weights=np.array([0.0, 0.0]),
        box_lower_constraints=np.array([-1.0, 0.5]),
        box_upper_constraints=np.array([1.0, 2.0]),
        eq_matrix=sp.csc_matrix(np.zeros((0, 2))),
        eq_bounds=np.array([]),
        neq_matrix=sp.csc_matrix(np.zeros((0, 2))),
        neq_lower_bounds=np.array([]),
        neq_upper_bounds=np.array([]),
    ), np.array([0.0, 0.5])


@pytest.fixture(scope="module")
def linear_weights_qp() -> tuple[QPData, np.ndarray]:
    """
    min_{x1, x2} 0.5 * (2*x1^2 + 2*x2^2) + 2x1 - 4x2
    s.t.
        -inf <= x1 <= inf
        -inf <= x2 <= inf

    expected solution:
        [-1.0, 2.0]
    """
    return QPData(
        quadratic_weights=np.array([2.0, 2.0]),
        linear_weights=np.array([2.0, -4.0]),
        box_lower_constraints=np.array([-np.inf, -np.inf]),
        box_upper_constraints=np.array([np.inf, np.inf]),
        eq_matrix=sp.csc_matrix(np.zeros((0, 2))),
        eq_bounds=np.array([]),
        neq_matrix=sp.csc_matrix(np.zeros((0, 2))),
        neq_lower_bounds=np.array([]),
        neq_upper_bounds=np.array([]),
    ), np.array([-1.0, 2.0])


@pytest.fixture(scope="module")
def infeasible_qp() -> QPData:
    """
    min_{x1} x1^2
    s.t.
        x1 = 1.0
        x1 = 2.0 (infeasible)
    """
    return QPData(
        quadratic_weights=np.array([1.0]),
        linear_weights=np.array([0.0]),
        box_lower_constraints=np.array([-np.inf]),
        box_upper_constraints=np.array([np.inf]),
        eq_matrix=sp.csc_matrix(np.array([[1.0], [1.0]])),
        eq_bounds=np.array([1.0, 2.0]),
        neq_matrix=sp.csc_matrix(np.zeros((0, 1))),
        neq_lower_bounds=np.array([]),
        neq_upper_bounds=np.array([]),
    )


@pytest.fixture(scope="module")
def larger_qp() -> tuple[QPData, np.ndarray]:
    """
    min_{x1, x2, x3} 0.5 * (2*x1^2 + 2*x2^2 + 2*x3^2)
    s.t.
        x1 + x2 + x3 = 1

    expected solution:
        [1/3, 1/3, 1/3]
    """
    return QPData(
        quadratic_weights=np.array([2.0, 2.0, 2.0]),
        linear_weights=np.array([0.0, 0.0, 0.0]),
        box_lower_constraints=np.array([-np.inf, -np.inf, -np.inf]),
        box_upper_constraints=np.array([np.inf, np.inf, np.inf]),
        eq_matrix=sp.csc_matrix(np.array([[1.0, 1.0, 1.0]])),
        eq_bounds=np.array([1.0]),
        neq_matrix=sp.csc_matrix(np.zeros((0, 3))),
        neq_lower_bounds=np.array([]),
        neq_upper_bounds=np.array([]),
    ), np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


def test_qp_data_inequality(simple_inequality_qp):
    qp_data, expected = simple_inequality_qp
    result = QPSolverQPSwift().solver_call(qp_data)
    assert np.allclose(result, expected)


def test_qp_data_simple_eq_as_inequality_qp(simple_eq_as_inequality_qp):
    qp_data, expected = simple_eq_as_inequality_qp
    result = QPSolverQPSwift().solver_call(qp_data)
    assert np.allclose(result, expected)


def test_qp_data_equality(simple_equality_qp):
    qp_data, expected = simple_equality_qp
    result = QPSolverQPSwift().solver_call(qp_data)
    assert np.allclose(result, expected)


def test_qp_data_box_constraints(box_constraints_qp):
    qp_data, expected = box_constraints_qp
    result = QPSolverQPSwift().solver_call(qp_data)
    assert np.allclose(result, expected)


def test_qp_data_linear_weights(linear_weights_qp):
    qp_data, expected = linear_weights_qp
    result = QPSolverQPSwift().solver_call(qp_data)
    assert np.allclose(result, expected)


def test_qp_data_larger_qp(larger_qp):
    qp_data, expected = larger_qp
    result = QPSolverQPSwift().solver_call(qp_data)
    assert np.allclose(result, expected, atol=1e-5)


def test_C_conditioning(simple_inequality_qp):
    qp_data, expected = simple_inequality_qp
    conditioning = Conditioning(C=sp.diags([69.0, 23.0]))
    conditioned_qp_data = conditioning.apply(qp_data)
    conditioned_result = QPSolverQPSwift().solver_call(conditioned_qp_data)
    result = conditioning.unapply(conditioned_result)
    assert np.allclose(result, expected)


def test_R_conditioning(simple_equality_qp):
    qp_data, expected = simple_equality_qp
    conditioning = Conditioning(R_eq=sp.diags([23.0]))
    conditioned_qp_data = conditioning.apply(qp_data)
    conditioned_result = QPSolverQPSwift().solver_call(conditioned_qp_data)
    result = conditioning.unapply(conditioned_result)
    assert np.allclose(result, expected)


def test_R_neq_conditioning(simple_inequality_qp):
    qp_data, expected = simple_inequality_qp
    conditioning = Conditioning(R_neq=sp.diags([10.0]))
    conditioned_qp_data = conditioning.apply(qp_data)
    conditioned_result = QPSolverQPSwift().solver_call(conditioned_qp_data)
    result = conditioning.unapply(conditioned_result)
    assert np.allclose(result, expected)


def test_combined_conditioning(simple_inequality_qp):
    qp_data, expected = simple_inequality_qp
    conditioning = Conditioning(C=sp.diags([2.0, 0.5]), R_neq=sp.diags([10.0]))
    conditioned_qp_data = conditioning.apply(qp_data)
    conditioned_result = QPSolverQPSwift().solver_call(conditioned_qp_data)
    result = conditioning.unapply(conditioned_result)
    assert np.allclose(result, expected)


def test_identity_conditioning(simple_inequality_qp):
    qp_data, expected = simple_inequality_qp
    conditioning = Conditioning(C=sp.diags([1.0, 1.0]), R_neq=sp.diags([1.0]))
    conditioned_qp_data = conditioning.apply(qp_data)
    conditioned_result = QPSolverQPSwift().solver_call(conditioned_qp_data)
    result = conditioning.unapply(conditioned_result)
    assert np.allclose(result, expected)


def test_no_conditioning(simple_inequality_qp):
    qp_data, expected = simple_inequality_qp
    conditioning = Conditioning()
    conditioned_qp_data = conditioning.apply(qp_data)
    conditioned_result = QPSolverQPSwift().solver_call(conditioned_qp_data)
    result = conditioning.unapply(conditioned_result)
    assert np.allclose(result, expected)
