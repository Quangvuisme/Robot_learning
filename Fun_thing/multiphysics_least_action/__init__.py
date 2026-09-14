"""Multiphysics as Optimization --- fluid + rigid body from one least action.

    minimize   int L^f(q^f, v^f) + L^r(q^r, v^r) + F^T q^f dt
    subject to div(v^f) = 0,  c(v^f, v^r) = 0

Variational calculus on that single problem yields incompressible Navier-Stokes
on one side, the manipulator equation on the other, and the coupling forces in
between --- see `coupling.py`, which implements exactly that.
"""

from .fluid import MACGrid
from .bodies import PendulumPaddle, FreePlate
from .coupling import unified_projection
from .sim import Simulation

__all__ = ["MACGrid", "PendulumPaddle", "FreePlate",
           "unified_projection", "Simulation"]
