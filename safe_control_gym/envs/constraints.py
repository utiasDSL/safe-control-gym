"""Constraints module.

Classes for constraints and lists of constraints.
"""

from __future__ import annotations

from gymnasium import Space, spaces
import sys

from typing import Callable
from enum import Enum
import numpy as np
import numpy.typing as npt


class ConstrainedVariableType(str, Enum):
    """Allowable constraint type specifiers."""

    STATE = "state"  # Constraints who are a function of the state X
    INPUT = "input"  # Constraints who are a function of the input U
    INPUT_AND_STATE = "input_and_state"  # Constraints who are a function of the input U and state X


class Constraint:
    """A (state-wise/trajectory-wise/stateful) constraint.

    A constraint can contain multiple scalar-valued constraint functions. Each should be represented
    as g(x) <= 0.
    """

    def __init__(
        self,
        state_space: Space,
        input_space: Space,
        ctype: ConstrainedVariableType,
        strict: bool = False,
        active_dims: list[int] | None = None,
        tolerance: npt.NDArray[np.float64] | None = None,
        rounding: int = 8,
    ):
        """Define the bounds and states.

        Args:
            state_space: System state space.
            input_space: System input space.
            ctype: Type of constraint (state, input, or both).
            strict: Option to check for strict constraint satisfaction at the threshold (< vs <=).
            active_dims: Optional list of indices to filter which dimensions are active.
            tolerance: The distance from the constraint at which is_almost_active returns True.
            rounding: Decimal places used in the `value()` method.

        """
        self.rounding = rounding
        if ctype == ConstrainedVariableType.STATE:
            dim = spaces.flatdim(state_space)
        elif ctype == ConstrainedVariableType.INPUT:
            dim = spaces.flatdim(input_space)
        elif ctype == ConstrainedVariableType.INPUT_AND_STATE:
            dim = spaces.flatdim(state_space) + spaces.flatdim(input_space)
        self.strict = strict
        self.ctype = ctype
        # Only want to select specific dimensions, implemented via a filter matrix.
        self.constraint_filter = np.eye(dim)
        if active_dims is not None:
            assert isinstance(active_dims, (list, np.ndarray)), "Active_dims is not a list/array."
            assert len(active_dims) <= dim, "More active_dim than constrainable self.dim"
            assert all(isinstance(n, int) for n in active_dims), "Non-integer active_dim."
            assert max(active_dims) < dim, "active_dim not stricly smaller than dim."
            assert len(active_dims) == len(set(active_dims)), "Duplicates in active_dim"
            self.constraint_filter = self.constraint_filter[active_dims]
            dim = len(active_dims)
        self.dim = dim
        self.n_constraints = dim
        self.tolerance = None if tolerance is None else np.array(tolerance, ndmin=1)

    def reset(self):
        """Clears up the constraint state (if any)."""

    def symbolic(self):
        """Create the symbolic form of the constraint function."""
        raise NotImplementedError

    def value(
        self,
        state: npt.NDArray[np.float64] | None = None,
        input: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Calculate the constraint function value.

        Args:
            state: The system state to evaluate the constraint at.
            input: The system input to evaluate the constraint at.

        Returns:
            The evaluation of the constraint.
        """
        if state is None and input is None:
            raise ValueError("Either state or input must be provided.")
        if self.ctype == ConstrainedVariableType.INPUT_AND_STATE:
            x = np.concatenate((state, input))
        elif self.ctype == ConstrainedVariableType.STATE:
            x = state
        else:
            x = input
        return np.round_(
            np.atleast_1d(np.squeeze(self.sym_func(np.array(x, ndmin=1)))),
            decimals=self.rounding,
        )

    def is_violated(
        self,
        state: npt.NDArray[np.float64] | None = None,
        input: npt.NDArray[np.float64] | None = None,
        c_value: npt.NDArray[np.float64] | None = None,
    ) -> bool:
        """Check if the constraint is violated.

        Args:
            state: The system state to evaluate the constraint at.
            input: The system input to evaluate the constraint at.
            c_value: Optional precomputed constraint value.

        Returns:
            bool: Whether the constraint was violeted.

        """
        c_value = self.value(state=state, input=input) if c_value is None else c_value
        if self.strict:
            return np.any(np.greater_equal(c_value, 0.0))
        return np.any(np.greater(c_value, 0.0))

    def is_almost_active(
        self,
        state: npt.NDArray[np.float64] | None = None,
        input: npt.NDArray[np.float64] | None = None,
        c_value: npt.NDArray[np.float64] | None = None,
    ) -> bool:
        """Check if the constraint is nearly violated.

        Can be used for reward shaping/constraint penalty in RL methods.

        Args:
            state: The system state to evaluate the constraint at.
            input: The system input to evaluate the constraint at.
            c_value: Optional precomputed constraint value.
        """
        if self.tolerance is None:
            return False
        c_value = self.value(state=state, input=input) if c_value is None else c_value
        return np.any(np.greater(c_value + self.tolerance, 0.0))

    def check_tolerance_shape(self):
        if self.tolerance is None:
            return
        if len(self.tolerance) != self.n_constraints:
            raise ValueError("tolerance dimension does not match the number of constraints.")


class QuadraticContstraint(Constraint):
    """Constraint class for constraints of the form x.T @ P @ x <= b."""

    def __init__(
        self,
        state_space,
        input_space,
        ctype: ConstrainedVariableType,
        P: np.ndarray,
        b: float,
        strict: bool = False,
        active_dims: list[int] | None = None,
        tolerance: list[float] | None = None,
    ):
        """Initializes the class.

        Args:
            state_space: System state space.
            input_space: System input space.
            ctype: Type of constraint (state, input, or both).
            P: The square matrix representing the quadratic.
            b: The scalar limit for the quadatic constraint.
            strict: Option to check for strict constraint satisfaction at the threshold (< vs <=).
            active_dims: Filters the constraint to only act only select certian dimensions.
            tolerance: The distance from the constraint at which is_almost_active returns True.

        """
        super().__init__(
            state_space,
            input_space,
            ctype,
            strict=strict,
            active_dims=active_dims,
            tolerance=tolerance,
        )
        P = np.array(P, ndmin=1)
        assert P.shape == (self.dim, self.dim), "P has the wrong dimension!"
        self.P = P
        assert isinstance(b, float), "b is not a scalar!"
        self.b = b
        self.n_constraints = 1  # Always scalar.
        self.sym_func = (
            lambda x: x.T @ self.constraint_filter.T @ self.P @ self.constraint_filter @ x - self.b
        )
        self.check_tolerance_shape()

    def symbolic(self):
        """Gets the symbolic form of the constraint function.

        Returns:
            lambda: The symbolic form of the constraint.

        """
        return self.sym_func


class LinearConstraint(Constraint):
    """Constraint class for constraints of the form A @ x <= b."""

    def __init__(
        self,
        state_space: Space,
        input_space: Space,
        ctype: ConstrainedVariableType,
        A: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        strict: bool = False,
        active_dims=None,
        tolerance=None,
    ):
        """Initialize the class.

        Args:
            state_space: System state space.
            input_space: System input space.
            ctype: Type of constraint (state, input, or both).
            A: A matrix of the constraint (self.n_constraints by self.dim).
            b: b matrix of the constraint (1D array self.n_constraints).
            strict: Option to check for strict constraint satisfaction at the threshold (< vs <=).
            active_dims: Filters the constraint to only act only select certian dimensions.
            tolerance: The distance from the constraint at which is_almost_active returns True.
        """
        super().__init__(
            state_space,
            input_space,
            ctype,
            strict=strict,
            active_dims=active_dims,
            tolerance=tolerance,
        )
        A = np.array(A, ndmin=1)
        b = np.array(b, ndmin=1)
        assert A.shape[1] == self.dim, "A has the wrong dimension!"
        self.A = A
        assert b.shape[0] == A.shape[0], "Dimension 0 of b does not match A!"
        self.b = b
        self.n_constraints = A.shape[0]
        self.sym_func = lambda x: self.A @ self.constraint_filter @ x - self.b
        self.check_tolerance_shape()

    def symbolic(self):
        """Gets the symbolic form of the constraint function.

        Returns:
            lambda: The symbolic form of the constraint.

        """
        return self.sym_func


class BoundedConstraint(LinearConstraint):
    """Class for bounded constraints lb <= x <= ub as polytopic constraints -Ix + b <= 0 and Ix - b <= 0."""

    def __init__(
        self,
        state_space: Space,
        input_space: Space,
        ctype: ConstrainedVariableType,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        strict: bool = False,
        active_dims: list[int] | None = None,
        tolerance: float | None = None,
    ):
        """Initialize the constraint.

        Args:
            state_space: System state space.
            input_space: System input space.
            ctype: Type of constraint (state, input, or both).
            lower_bounds (np.array or list): Lower bound of constraint.
            upper_bounds (np.array or list): Uppbound of constraint.
            constrained_variable (ConstrainedVariableType): Type of constraint.
            strict: Option to check for strict constraint satisfaction at the threshold (< vs <=).
            active_dims: List specifying which dimensions the constraint is active for.
            tolerance: The distance at which is_almost_active(env) triggers.

        """
        self.lower_bounds = np.array(lower_bounds, ndmin=1)
        self.upper_bounds = np.array(upper_bounds, ndmin=1)
        dim = self.lower_bounds.shape[0]
        A = np.vstack((-np.eye(dim), np.eye(dim)))
        b = np.hstack((-self.lower_bounds, self.upper_bounds))
        super().__init__(
            state_space,
            input_space,
            ctype,
            A,
            b,
            strict=strict,
            active_dims=active_dims,
            tolerance=tolerance,
        )
        self.check_tolerance_shape()


class DefaultConstraint(BoundedConstraint):
    """Use the environment's observation_space or action_space for default state or input bound constraints.

    This class only constraint either STATE or INPUT constraint but not both. The class constrains
    the entire variable, i.e. no `active_dims` option. For other options, use BoundedConstraint.
    """

    def __init__(
        self,
        state_space: Space,
        input_space: Space,
        ctype: ConstrainedVariableType,
        lower_bounds: npt.NDArray[np.float64] | None = None,
        upper_bounds: npt.NDArray[np.float64] | None = None,
        strict: bool = False,
        tolerance: float | None = None,
    ):
        """ "Initialize the class.

        Args:
            state_space: System state space.
            input_space: System input space.
            ctype: Type of constraint (state, input, or both).
            lower_bounds: 1D array of the lower bounds. Length must match the space dimension. If
                None, the env defaults are used.
            upper_bounds: 1D array of the lower bounds. Length must match the space dimension. If
                None, the env defaults are used.
            strict: Option to check for strict constraint satisfaction at the threshold (< vs <=).
            tolerance: The distance at which is_almost_active(env) triggers.
        """
        if ctype == ConstrainedVariableType.STATE:
            cspace = spaces.flatten_space(state_space)
        elif ctype == ConstrainedVariableType.INPUT:
            cspace = spaces.flatten_space(input_space)
        else:
            raise NotImplementedError("DefaultConstraint can only be of type STATE or INPUT")
        upper_bounds = cspace.high if upper_bounds is None else np.array(upper_bounds, ndmin=1)
        lower_bounds = cspace.low if lower_bounds is None else np.array(lower_bounds, ndmin=1)
        assert len(upper_bounds) == cspace.shape[0], "Upper bound does not match space dimension."
        assert len(lower_bounds) == cspace.shape[0], "Lower bound does not match space dimension."
        super().__init__(
            state_space,
            input_space,
            ctype,
            lower_bounds=lower_bounds.astype(np.float64),
            upper_bounds=upper_bounds.astype(np.float64),
            strict=strict,
            active_dims=None,
            tolerance=tolerance,
        )


class ConstraintList:
    """Collates a (ordered) list of constraints."""

    def __init__(self, constraints: list[Constraint]):
        """Initialize the constraint list.

        Args:
            constraints: A list of constraints.
        """
        self.constraints = constraints
        constraint_lengths = [con.n_constraints for con in self.constraints]
        # 1st constraint is always index 0, hence ignored
        self.constraint_indices = np.cumsum(constraint_lengths[:-1])
        self.n_constraints = sum(constraint_lengths)
        # Constraint subsets
        self.state_constraints = [
            con for con in self.constraints if con.ctype == ConstrainedVariableType.STATE
        ]
        self.input_constraints = [
            con for con in self.constraints if con.ctype == ConstrainedVariableType.INPUT
        ]
        self.input_state_constraints = [
            con for con in self.constraints if con.ctype == ConstrainedVariableType.INPUT_AND_STATE
        ]

    def __len__(self) -> int:
        """Get the constraint list length."""
        return len(self.constraints)

    def symbolic(self, state_models: bool = True, input_models: bool = True) -> list[Callable]:
        """Return all the symbolic models the constraints."""
        assert state_models or input_models, "Select at least one of state_models or input_models."
        if state_models and input_models:
            return [con.symbolic() for con in self.constraints]
        if state_models:
            return [con.symbolic() for con in self.state_constraints]
        return [con.symbolic() for con in self.input_constraints]

    def value(
        self,
        state: npt.NDArray[np.float64] | None = None,
        input: npt.NDArray[np.float64] | None = None,
        only_state: bool = False,
    ):
        """Get all constraint function values."""
        if self.n_constraints == 0:
            return np.array([])
        if only_state:
            return np.concatenate([con.value(state) for con in self.state_constraints])
        return np.concatenate([con.value(state, input) for con in self.constraints])

    def is_violated(
        self,
        state: npt.NDArray[np.float64] | None = None,
        input: npt.NDArray[np.float64] | None = None,
        c_value=None,
    ):
        """Check if any of the constraints is violated."""
        if c_value is not None:
            splits = np.split(c_value, self.constraint_indices)
            return any(
                con.is_violated(state, input, c_value=split)
                for con, split in zip(self.constraints, splits)
            )
        return any(con.is_violated(state, input) for con in self.constraints)

    def is_almost_active(
        self,
        state: npt.NDArray[np.float64] | None = None,
        input: npt.NDArray[np.float64] | None = None,
        c_value=None,
    ):
        """Check if constraint is nearly violated."""
        if c_value is not None:
            splits = np.split(c_value, self.constraint_indices)
            return any(
                con.is_almost_active(state, input, c_value=split)
                for con, split in zip(self.constraints, splits)
            )
        return any(con.is_almost_active(state, input) for con in self.constraints)

    @staticmethod
    def from_specs(
        state_space: Space, action_space: Space, constraint_specs: list[dict]
    ) -> ConstraintList:
        """Creates a ConstraintList from constraint specification.

        Args:
            state_space: System state space.
            input_space: System input space.
            constraint_specs: List of dicts defining the constraints info.
        """
        constraint_list = []
        for constraint in constraint_specs:
            assert isinstance(constraint, dict), "Each constraint must be specified as a dict."
            assert "type" in constraint.keys(), "Each constraint must have a 'type' key"
            c_class = getattr(sys.modules[__name__], constraint["type"])
            kwargs = {k: v for k, v in constraint.items() if k != "type"}
            constraint_list.append(c_class(state_space, action_space, **kwargs))
        return ConstraintList(constraint_list)
