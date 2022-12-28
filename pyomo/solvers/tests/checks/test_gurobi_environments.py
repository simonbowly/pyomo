"""
These tests are intended to be run with a single use license.
The failure types are equivalent for any resource contention in Gurobi
environments (compute server sessions, token server licenses, instant
cloud, WLS tokens)
"""

import gc
from contextlib import contextmanager

import pytest
import gurobipy as gp

import pyomo.common.errors as pyo_errors
import pyomo.environ as pyo
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect


def cleanup():
    GurobiDirect._verified_license = None
    GurobiDirect._import_messages = ""
    GurobiDirect._name = None
    GurobiDirect._version = 0
    GurobiDirect._version_major = 0
    gc.collect()
    gp.disposeDefaultEnv()


def using_singleuse_license():
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
    """If the GurobiDirect.available() check fails to create a model, it stores
    the error message in global state and always returns it in future. For single
    use or token licenses, any failure renders the application unusable.

    Ideally, available() should just check if gurobipy is importable, it
    shouldn't try to start an environment.

    Fails with:

        pyomo.common.errors.ApplicationError: Could not create a gurobipy Model
        for <class 'pyomo.solvers.plugins.solvers.gurobi_direct.GurobiDirect'>
        solver plugin

    """
    with pyomo_global_cleanup():
        model = pyo.ConcreteModel()
        with gp.Env():
            opt = pyo.SolverFactory("gurobi_direct")
            try:
                opt.solve(model)
            except pyo_errors.ApplicationError:
                # Expected failure for a single use license. There is another
                # environment open so the default env cannot be started.
                pass
        # This not raise an error, since the other environment has been freed.
        opt.solve(model)


@pytest.mark.solver("gurobi")
@pytest.mark.skipif(
    not using_singleuse_license(), reason="test needs a single use license"
)
def test_set_environment_options():
    """
    Cannot set environment-only options since pyomo only deals with Models.

    Fails with:

        gurobipy.GurobiError: Unable to modify parameter MemLimit after
        environment started

    Options:
    1. Identify the environment parameters and apply them separately?
    2. Nested options in the dictionary?
    3. Separate argument env_options to SolverFactory?

    Does this need to be compatible with the command line way of calling pyomo?

    """
    with pyomo_global_cleanup():
        model = pyo.ConcreteModel()
        opt = pyo.SolverFactory(
            "gurobi_direct", options={"ComputeServer": "/url/to/server"}
        )
        # Check that the error comes from an attempted connection, not from setting
        # the parameter after the environment is started.
        with pytest.raises(pyo_errors.ApplicationError, match="Could not resolve host"):
            opt.solve(model)


@pytest.mark.solver("gurobi")
@pytest.mark.skipif(
    not using_singleuse_license(), reason="test needs a single use license"
)
def test_environment_context():
    """
    The context management feature of pyomo should be used to correctly
    dispose of environments.
    """
    with pyomo_global_cleanup():
        with pyo.SolverFactory("gurobi_direct") as opt:
            model = pyo.ConcreteModel()
            opt.solve(model)
        # Ideally the environment and all models would be freed at this point.
        # It could be made implicit, in the sense that if you use the context
        # manager form, environment creation is triggerred by __enter__ and
        # disposed in __exit__, but if you don't use the class as a context manager
        # then the default env is used?
        with gp.Env():
            pass


@pytest.mark.solver("gurobi")
@pytest.mark.skipif(
    not using_singleuse_license(), reason="test needs a single use license"
)
def test_default_environment_disposal():
    """
    This fails because there is no public API for cleaning up the model and it
    doesn't fall out of scope.
    """
    with pyomo_global_cleanup():
        with pyo.SolverFactory("gurobi_direct") as opt:
            model = pyo.ConcreteModel()
            opt.solve(model)
        gp.disposeDefaultEnv()
        # opt is still in scope, as is it's _solver_model. This also needs to be
        # disposed of or the default env is not really freed.
        with gp.Env():
            pass


@pytest.mark.xfail
@pytest.mark.solver("gurobi")
@pytest.mark.skipif(
    not using_singleuse_license(), reason="test needs a single use license"
)
def test_multiple_solvers():
    """This currently passes but would fail if each of opt1 and opt2 created their
    own environments by default (and they were used without context managers).
    Something to be careful of."""

    with pyomo_global_cleanup():

        opt1 = pyo.SolverFactory("gurobi_direct")
        model1 = pyo.ConcreteModel()
        opt1.solve(model1)

        opt2 = pyo.SolverFactory("gurobi_direct")
        model2 = pyo.ConcreteModel()
        opt2.solve(model2)


@pytest.mark.solver("gurobi")
@pytest.mark.skipif(
    not using_singleuse_license(), reason="test needs a single use license"
)
def test_multiple_models_leaky():
    """Kind of a silly test, but this is just to point out that GurobiDirect
    creates a new _solver_model with each solve() call, and doesn't dispose
    the old one. If the scope accidentally leaks then a license or remote
    environment can remain in use."""

    with pyomo_global_cleanup():

        opt = pyo.SolverFactory("gurobi_direct")

        model1 = pyo.ConcreteModel()
        opt.solve(model1)

        tmp = opt._solver_model

        model2 = pyo.ConcreteModel()
        opt.solve(model2)

        opt._solver_model.dispose()
        del opt
        gp.disposeDefaultEnv()

        # Still holding a leaked model reference in tmp, so the environment
        # is not properly freed.
        with gp.Env():
            pass
