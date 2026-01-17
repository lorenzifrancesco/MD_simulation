import numpy as np
from Heating  import *
from Beams import *

from tqdm import trange

def _coerce_state(x0, v0):
    x0 = np.asarray(x0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    if x0.ndim == 1:
        x0 = x0[:, None]
    if v0.ndim == 1:
        v0 = v0[:, None]
    if x0.ndim != 2 or v0.ndim != 2:
        raise ValueError("x0 and v0 must have shape (D, N)")
    if x0.shape != v0.shape:
        raise ValueError("x0 and v0 must have the same shape")
    return x0, v0


def _active_mask(xs, z_index):
    if z_index is None:
        return np.ones(xs.shape[1], dtype=bool)
    return xs[z_index] > 0.0


def _below_zmin(xs, z_index, z_min):
    if z_index is None:
        return False
    return np.min(xs[z_index]) < z_min


def Heating_1(xs, vs, dt, i, beam=GaussianBeam()):
    sigma_i_rho, sigma_i_zeta = np.std(vs[i], axis=1) # std across atoms
    r_atoms = xs[i]
    dsigma_i = dsigma_v(r_atoms, dt, beam)

    sigma_i_rho = sigma_i_rho * beam.vs_rho
    sigma_i_zeta = sigma_i_zeta * beam.vs_zeta
    
    new_sigma_rho = (sigma_i_rho + dsigma_i) / beam.vs_rho
    new_sigma_zeta = (sigma_i_zeta + dsigma_i) / beam.vs_zeta
    
    vs[i, 0, :] = vs[i, 0, :] + np.random.normal(0, scale=new_sigma_rho, size=len(vs[i, 0, :]))
    vs[i, 1, :] = vs[i, 1, :] + np.random.normal(0, scale=new_sigma_zeta, size=len(vs[i, 0, :]))

def Heating_2(xs, vs, dt, i, beam=GaussianBeam()):
    r_atoms = xs[i]
    v_atoms = vs[i]
    new_velocity = AddScattering(r_atoms, v_atoms, dt, beam) 
    return new_velocity


def verlet(
    x0,
    v0,
    a_func,
    dt,
    N_steps,
    N_saves,
    beam=GaussianBeam(),
    HEATING=False,
    progress=True,
    z_index=1,
):
    """
    Integrate atomic motion using the velocity-Verlet scheme, storing only
    N_saves snapshots instead of the full N_steps+1 history.

    The integrator evolves atomic trajectories in dimensionless units under the
    acceleration field provided by `a_func`. This implementation assumes that
    the optical potential is zero for ζ < 0 (atoms past the fiber tip).

    Parameters
    ----------
    x0 : (D,) array_like or (D, N) ndarray
        Initial positions in dimensionless units for D coordinates.
        For cylindrical runs: x[0] = rho, x[1] = zeta.
    v0 : (D,) array_like or (D, N) ndarray
        Initial velocities in dimensionless units.
    a_func : callable
        Function of the form `a_func(x)` returning acceleration components
        Must return accelerations with the same shape as `x`.
    dt : float
        Time step (dimensionless units).
    N_steps : int
        Number of integration steps.
    N_saves : int
        Number of snapshots to store (including t=0 and final time).
    beam : Beam, optional
        Beam object used for heating (if enabled).
    HEATING : bool, optional
        If True, apply Heating_2 at each step (default: False).
    progress : bool, optional
        If True, display a tqdm progress bar (default: True).
    z_index : int or None, optional
        Index of the coordinate used for the z > 0 boundary condition.
        Set to None to disable boundary masking (default: 1).

    Returns
    -------
    xs : ndarray, shape (N_saves, D, N)
        Atomic positions at saved time steps.
    vs : ndarray, shape (N_saves, D, N)
        Atomic velocities at saved time steps.
    ts : ndarray, shape (N_saves,)
        Dimensionless time values at saved time steps.

    Notes
    -----
    - Uses a Taylor expansion for the first step.
    - Enforces boundary condition: motion only where z_index > 0.
    - Velocities are estimated via central differences for internal steps
      and via a forward difference for the final step (as in the original).
    """

    x0, v0 = _coerce_state(x0, v0)
    dim, N = x0.shape

    # Decide which step indices we will save:
    # steps from 0..N_steps, we want N_saves snapshots spread across them.
    # step index k corresponds to time t = k * dt
    save_indices = np.linspace(0, N_steps, N_saves, dtype=int)
    save_indices = np.unique(save_indices)  # just in case we have repeated indices

    # Pre-allocate only the saved snapshots
    xs_save = np.zeros((len(save_indices), dim, N), dtype=float)
    vs_save = np.zeros_like(xs_save)
    ts_save = np.zeros(len(save_indices), dtype=float)

    # Rolling buffers
    xs_prev = x0.copy()
    v_prev = v0.copy()
    
    a0 = a_func(xs_prev)

    xs_curr = xs_prev + v_prev * dt + 0.5 * a0 * dt * dt
    v_curr = v_prev.copy() # in absence of data for the central difference

    t = 0.0
    step = 0

    save_ptr = 0
    next_save_step = save_indices[save_ptr]

    # Save initial state (step=0, t=0)
    if step == next_save_step:
        xs_save[save_ptr] = xs_prev
        vs_save[save_ptr] = v_prev
        ts_save[save_ptr] = t
        save_ptr += 1
        if save_ptr < len(save_indices):
            next_save_step = save_indices[save_ptr]

    iterator = trange(1, N_steps, desc="Simulation", disable=not progress, mininterval=1.0)
    for step in iterator:
        # At the start of this iteration:
        #   xs_prev = x_{step-1}
        #   xs_curr = x_{step}
        # We will compute x_{step+1} and v_{step} via central difference.

        # Boundary mask (only atoms with z > 0 evolve)
        update = _active_mask(xs_curr, z_index)

        a = a_func(xs_curr)

        xs_next = xs_curr + (xs_curr - xs_prev + a * dt * dt) * update

        # Velocity at time t_step via centered difference:
        v_curr = (xs_next - xs_prev) / (2.0 * dt) * update

        if HEATING:
            # Heating_2 expects (time, 2, N); fake a single-step trajectory
            xs_fake = xs_curr[None, ...]
            vs_fake = v_curr[None, ...]
            v_curr = Heating_2(xs_fake, vs_fake, dt, 0, beam) # TODO change func signature to make the calling more natural

        t = step * dt  # xs_curr corresponds to time t
        # Save snapshot corresponding to xs_curr, v_curr, t
        if step == next_save_step and save_ptr < len(save_indices):
            xs_save[save_ptr] = xs_curr
            vs_save[save_ptr] = v_curr
            ts_save[save_ptr] = t
            save_ptr += 1
            if save_ptr < len(save_indices):
                next_save_step = save_indices[save_ptr]

        xs_prev, xs_curr = xs_curr, xs_next
        v_prev = v_curr

    # After the loop, xs_curr holds x_{N_steps}, xs_prev holds x_{N_steps-1}
    # We compute the final velocity via forward difference (as in your original code):
    update_final = _active_mask(xs_curr, z_index)
    v_final = (xs_curr - xs_prev) / dt * update_final

    # Save final state if N_steps is among requested save_indices
    final_step = N_steps
    t_final = final_step * dt
    if save_ptr < len(save_indices) and final_step == save_indices[save_ptr]:
        xs_save[save_ptr] = xs_curr
        vs_save[save_ptr] = v_final
        ts_save[save_ptr] = t_final
        save_ptr += 1

    return xs_save, vs_save, ts_save


def verlet_up_to(
    x0,
    v0,
    a_func,
    dt,
    N_steps,
    z_min=5.0,
    HEATING=False,
    beam=None,
    z_index=1,
):
    """
    Integrate atomic motion using a position-form velocity-Verlet scheme
    until the minimum z coordinate drops below z_min or N_steps are done.

    Parameters
    ----------
    x0 : (D,) array_like or (D, N) ndarray
        Initial positions in dimensionless units.
    v0 : (D,) array_like or (D, N) ndarray
        Initial velocities in dimensionless units.
    a_func : callable
        Function of the form `a_func(x)` returning acceleration components
        Must return accelerations with the same shape as `x`.
    dt : float
        Time step (dimensionless units).
    N_steps : int
        Maximum number of integration steps.
    z_min : float, optional
        Stopping condition: stop integration when any atom reaches z < z_min.
    HEATING : bool, optional
        If True, apply Heating_2 at each step (default: False).
    beam : Beam, optional
        Beam object to pass to Heating_2 if HEATING=True.
    z_index : int or None, optional
        Index of the coordinate used for the z_min condition.
        Set to None to disable early stopping (default: 1).

    Returns
    -------
    x_final : ndarray, shape (D, N)
        Final atomic positions.
    v_final : ndarray, shape (D, N)
        Final atomic velocities.
    """
    x0, v0 = _coerce_state(x0, v0)
    xs_prev = x0.copy()

    a0 = a_func(xs_prev)
    xs_curr = xs_prev + v0 * dt + 0.5 * a0 * dt * dt
    v_curr = v0.copy()

    if _below_zmin(xs_curr, z_index, z_min):
        return xs_curr, v_curr

    for step in range(1, N_steps):
        a_curr = a_func(xs_curr)

        xs_next = xs_curr + (xs_curr - xs_prev + a_curr * dt * dt)
        v_curr = (xs_next - xs_prev) / (2.0 * dt)

        if HEATING:
            # Heating_2(xs, vs, dt, i, beam) only uses xs[i], vs[i]
            # so we fake a single-step trajectory with i=0
            xs_fake = xs_curr[None, ...]
            vs_fake = v_curr[None, ...]
            v_curr = Heating_2(xs_fake, vs_fake, dt, 0, beam)
        xs_prev, xs_curr = xs_curr, xs_next

        if _below_zmin(xs_curr, z_index, z_min):
            break
    return xs_curr, v_curr
