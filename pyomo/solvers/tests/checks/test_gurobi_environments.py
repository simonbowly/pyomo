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
def test_set_environment_options_notmanaged():
    """ Solver options should handle parameters which must be set before the
    environment is started (i.e. connection params, memory limits). If they
    are set without manage_env, then pyomo will try to set them on the model,
    which will fail."""
    with pyomo_global_cleanup():

        model = pyo.ConcreteModel()
        opt = pyo.SolverFactory(
            "gurobi_direct", manage_env=False,
            options={"ComputeServer": "/url/to/server"},
        )

        # Check that the error comes from an attempted connection, not from setting
        # the parameter after the environment is started.
        with pytest.raises(gp.GurobiError, match="Unable to modify"):
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


@pytest.mark.solver("gurobi")
def test_param_set():

    # Make sure parameters aren't set twice. If they are set on
    # the environment, they shouldn't be set on the model.

    with pyomo_global_cleanup():

        from unittest import mock

        envparams = {}
        modelparams = {}

        class TempEnv(gp.Env):
            def setParam(self, param, value):
                envparams[param] = value

        class TempModel(gp.Model):
            def setParam(self, param, value):
                modelparams[param] = value

        with mock.patch("gurobipy.Env", new=TempEnv), mock.patch("gurobipy.Model", new=TempModel):

            with pyo.SolverFactory("gurobi_direct", options={'Method': 2, 'MIPFocus': 1}, manage_env=True) as opt:
                model = pyo.ConcreteModel()
                opt.solve(model, options={'MIPFocus': 2})

        # Method should not be set again, but MIPFocus was changed
        assert envparams == {"Method": 2, "MIPFocus": 1}
        assert modelparams == {"MIPFocus": 2, "OutputFlag": 0}


@pytest.mark.solver("gurobi")
def test_param_changes():

    # Try an erroneous parameter setting to ensure parameters go through
    # FIXME: different exception classes depending on the method used

    with pyomo_global_cleanup():

        # Default env: parameters set on model at solve time

        with pyo.SolverFactory("gurobi_direct", options={'Method': 20}) as opt:
            model = pyo.ConcreteModel()
            with pytest.raises(gp.GurobiError, match='Unable to set'):
                opt.solve(model)

    with pyomo_global_cleanup():

        # Managed env: parameters set on env at solve time

        with pyo.SolverFactory("gurobi_direct", options={'Method': 20}, manage_env=True) as opt:
            model = pyo.ConcreteModel()
            with pytest.raises(pyo_errors.ApplicationError, match='Unable to set'):
                opt.solve(model)

    with pyomo_global_cleanup():

        # Managed env: parameters set on env at solve time

        opt = pyo.SolverFactory("gurobi_direct", options={'Method': 20}, manage_env=True)
        try:
            model = pyo.ConcreteModel()
            with pytest.raises(pyo_errors.ApplicationError, match='Unable to set'):
                opt.solve(model)
        finally:
            opt.close()

    with pyomo_global_cleanup():

        # Default env: parameters passed to solve()

        with pyo.SolverFactory("gurobi_direct") as opt:
            model = pyo.ConcreteModel()
            with pytest.raises(gp.GurobiError, match='Unable to set'):
                opt.solve(model, options={'Method': 20})

    with pyomo_global_cleanup():

        # Managed env: parameters passed to solve()

        with pyo.SolverFactory("gurobi_direct", manage_env=True) as opt:
            model = pyo.ConcreteModel()
            with pytest.raises(gp.GurobiError, match='Unable to set'):
                opt.solve(model, options={'Method': 20})
