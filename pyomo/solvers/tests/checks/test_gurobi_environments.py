"""
Tests for working with Gurobi environments. Some require a single-use license
and are skipped if this isn't the case.
"""

import gc
from contextlib import contextmanager

import pytest
import gurobipy as gp

import pyomo.common.errors as pyo_errors
import pyomo.environ as pyo
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect


def using_singleuse_license():
    # Return true if the current license is single-use
    cleanup()
    try:
        with gp.Env():
            try:
                with gp.Env():
                    # License allows multiple uses
                    return False
            except gp.GurobiError:
                return True
    except gp.GurobiError:
        # No license available
        return False


def cleanup():
    gc.collect()
    gp.disposeDefaultEnv()


@contextmanager
def pyomo_global_cleanup():
    """Forcefully clean up pyomo's global state after exiting this context."""
    cleanup()
    try:
        yield
    finally:
        cleanup()


@pytest.mark.solver("gurobi")
@pytest.mark.skipif(
    not using_singleuse_license(), reason="test needs a single use license"
)
def test_persisted_license_failure():
    """ Solver should allow retries to start the environment, instead of
    persisting the same failure. """
    with pyomo_global_cleanup():

        model = pyo.ConcreteModel()
        with gp.Env():
            opt = pyo.SolverFactory("gurobi_direct", manage_env=True)
            try:
                # Expected to fail: there is another environment open so the
                # solver env cannot be started.
                opt.solve(model)
                assert False  # Something wrong with the test if we got here
            except pyo_errors.ApplicationError:
                pass
        # Should not raise an error, since the other environment has been freed.
        opt.solve(model)
        opt.close()


@pytest.mark.solver("gurobi")
def test_set_environment_options():
    """ Solver options should handle parameters which must be set before the
    environment is started (i.e. connection params, memory limits). """
    with pyomo_global_cleanup():

        model = pyo.ConcreteModel()
        opt = pyo.SolverFactory(
            "gurobi_direct", manage_env=True,
            options={"ComputeServer": "/url/to/server"},
        )

        # Check that the error comes from an attempted connection, not from setting
        # the parameter after the environment is started.
        with pytest.raises(pyo_errors.ApplicationError, match="Could not resolve host"):
            opt.solve(model)


@pytest.mark.solver("gurobi")
@pytest.mark.skipif(
    not using_singleuse_license(), reason="test needs a single use license"
)
def test_context():
    """ Context management should close the gurobi environment. """
    with pyomo_global_cleanup():

        with pyo.SolverFactory("gurobi_direct", manage_env=True) as opt:
            model = pyo.ConcreteModel()
            opt.solve(model)

        # Environment closed, so another can be created
        with gp.Env():
            pass


@pytest.mark.solver("gurobi")
@pytest.mark.skipif(
    not using_singleuse_license(), reason="test needs a single use license"
)
def test_close():
    """ Manual close() method  should close the gurobi environment. """
    with pyomo_global_cleanup():

        opt = pyo.SolverFactory("gurobi_direct", manage_env=True)
        model = pyo.ConcreteModel()
        opt.solve(model)
        opt.close()

        # Environment closed, so another can be created
        with gp.Env():
            pass


@pytest.mark.solver("gurobi")
@pytest.mark.skipif(
    not using_singleuse_license(), reason="test needs a single use license"
)
def test_multiple_solvers():
    """ Breaking change: this would previously have worked since multiple
    solvers share the default environment. The workaround is easy (re-use
    the solver) but some users may have written their code this way. """

    with pyomo_global_cleanup():

        try:

            opt1 = pyo.SolverFactory("gurobi_direct")
            model1 = pyo.ConcreteModel()
            opt1.solve(model1)

            opt2 = pyo.SolverFactory("gurobi_direct")
            model2 = pyo.ConcreteModel()
            opt2.solve(model2)

        finally:

            opt1.close()
            opt2.close()
            gp.disposeDefaultEnv()


@pytest.mark.solver("gurobi")
@pytest.mark.skipif(
    not using_singleuse_license(), reason="test needs a single use license"
)
def test_multiple_models_leaky():
    """ Make sure all models are closed by the GurobiDirect instance. """

    with pyomo_global_cleanup():

        with pyo.SolverFactory("gurobi_direct", manage_env=True) as opt:

            model1 = pyo.ConcreteModel()
            opt.solve(model1)

            # Leak a model reference, then create a new model.
            # Pyomo should close the old model since it is no longed needed.
            tmp = opt._solver_model

            model2 = pyo.ConcreteModel()
            opt.solve(model2)

        # Context properly closed all models and environments
        with gp.Env():
            pass
