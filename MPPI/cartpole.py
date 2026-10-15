"""Cartpole dynamics, batched on the GPU with torch.

State x = [p, pdot, theta, thetadot]
    p        cart position on the rail          [m]
    theta    pole angle, 0 = hanging down, pi = upright   [rad]
Action u = force on the cart [N], applied as u = u_prev + eps (see MPPI).

Standard underactuated cartpole ODE (Barto/Sutton form, continuous time),
integrated with semi-implicit (symplectic) Euler for stability at the step
sizes MPPI uses.
"""

import torch


class CartpoleParams:
    def __init__(self, mc=1.0, mp=0.1, l=0.5, g=9.81, damping=0.02,
                 rail_limit=2.4, force_limit=15.0):
        self.mc = mc                # cart mass [kg]
        self.mp = mp                # pole mass [kg]
        self.l = l                  # half pole length [m] (distance to CoM)
        self.g = g
        self.damping = damping      # viscous damping at the pivot
        self.rail_limit = rail_limit
        self.force_limit = force_limit


def dynamics(x, u, dt, p: CartpoleParams):
    """One symplectic-Euler step.

    x: (..., 4) tensor [pos, vel, theta, thetadot]
    u: (...,)   tensor, force on the cart [N]
    Returns x_next with the same leading shape as x.
    """
    pos, vel, th, thd = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    u = u.clamp(-p.force_limit, p.force_limit)

    sin_th, cos_th = torch.sin(th), torch.cos(th)
    mt = p.mc + p.mp

    # standard cartpole equations of motion (theta measured from downward)
    num = (u + p.mp * p.l * thd ** 2 * sin_th - p.damping * thd * cos_th / p.l)
    denom = mt - p.mp * cos_th ** 2
    acc = num / denom
    thdd = (-p.g * sin_th - acc * cos_th - p.damping * thd / (p.mp * p.l ** 2)) / p.l

    # symplectic Euler: update velocities first, then positions with the NEW
    # velocities -- much more stable than explicit Euler for a pendulum
    vel_n = vel + dt * acc
    thd_n = thd + dt * thdd
    pos_n = pos + dt * vel_n
    th_n = th + dt * thd_n

    # hard walls at the rail limits: kill outward velocity, clamp position
    hit_pos = pos_n > p.rail_limit
    hit_neg = pos_n < -p.rail_limit
    pos_n = torch.where(hit_pos, torch.full_like(pos_n, p.rail_limit), pos_n)
    pos_n = torch.where(hit_neg, torch.full_like(pos_n, -p.rail_limit), pos_n)
    vel_n = torch.where(hit_pos | hit_neg, torch.zeros_like(vel_n), vel_n)

    return torch.stack([pos_n, vel_n, th_n, thd_n], dim=-1)


def angle_wrap(theta):
    """Wrap to (-pi, pi] so cost functions don't see a discontinuity at 2*pi."""
    return (theta + torch.pi) % (2 * torch.pi) - torch.pi
