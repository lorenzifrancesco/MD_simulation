import numpy as np
from Heating  import *
from Beams import *
from numba import njit, prange

from tqdm import trange

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
):
    """
    Integrate atomic motion using the velocity-Verlet scheme, storing only
    N_saves snapshots instead of the full N_steps+1 history.

    The integrator evolves atomic trajectories in dimensionless units under the
    acceleration field provided by `a_func`. This implementation assumes that
    the optical potential is zero for ζ < 0 (atoms past the fiber tip).

    Parameters
    ----------
    x0 : (2,) array_like or (2, N) ndarray
        Initial positions in dimensionless units:
        - x[0] = ρ (radial coordinate in w0 units)
        - x[1] = ζ (axial coordinate in zR units).
        Supports single-particle or multi-particle arrays.
    v0 : (2,) array_like or (2, N) ndarray
        Initial velocities in dimensionless units.
    a_func : callable
        Function of the form `a_func(x)` returning acceleration components
        (aρ, aζ) at position `x`. Must support vectorized input.
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

    Returns
    -------
    xs : ndarray, shape (N_saves, 2, N)
        Atomic positions at saved time steps.
    vs : ndarray, shape (N_saves, 2, N)
        Atomic velocities at saved time steps.
    ts : ndarray, shape (N_saves,)
        Dimensionless time values at saved time steps.

    Notes
    -----
    - Uses a Taylor expansion for the first step.
    - Enforces boundary condition: motion only for ζ > 0.
    - Velocities are estimated via central differences for internal steps
      and via a forward difference for the final step (as in the original).
    """

    x0 = np.asarray(x0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    N = x0.shape[1]

    # Decide which step indices we will save:
    # steps from 0..N_steps, we want N_saves snapshots spread across them.
    # step index k corresponds to time t = k * dt
    save_indices = np.linspace(0, N_steps, N_saves, dtype=int)
    save_indices = np.unique(save_indices)  # just in case we have repeated indices

    # Pre-allocate only the saved snapshots
    xs_save = np.zeros((len(save_indices), 2, N), dtype=float)
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
        z = xs_curr[1]
        update = z > 0.0

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
    z_final = xs_curr[1]
    update_final = z_final > 0.0
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
):
    """
    Integrate atomic motion using a position-form velocity-Verlet scheme
    until the minimum z coordinate drops below z_min or N_steps are done.

    Parameters
    ----------
    x0 : (2,) array_like or (2, N) ndarray
        Initial positions in dimensionless units:
        - x[0] = ρ (radial coordinate in w0 units)
        - x[1] = ζ (axial coordinate in zR units).
        Supports single-particle or multi-particle arrays.
    v0 : (2,) array_like or (2, N) ndarray
        Initial velocities in dimensionless units.
    a_func : callable
        Function of the form `a_func(x)` returning acceleration components
        (aρ, aζ) at position `x`. Must support vectorized input.
    dt : float
        Time step (dimensionless units).
    N_steps : int
        Maximum number of integration steps.
    z_min : float, optional
        Stopping condition: stop integration when any atom reaches ζ < z_min.
    HEATING : bool, optional
        If True, apply Heating_2 at each step (default: False).
    beam : Beam, optional
        Beam object to pass to Heating_2 if HEATING=True.

    Returns
    -------
    x_final : ndarray, shape (2, N)
        Final atomic positions.
    v_final : ndarray, shape (2, N)
        Final atomic velocities.
    """
    x0 = np.asarray(x0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    xs_prev = x0.copy()

    a0 = a_func(xs_prev)
    xs_curr = xs_prev + v0 * dt + 0.5 * a0 * dt * dt
    v_curr = v0.copy()

    if np.min(xs_curr[1]) < z_min:
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

        if np.min(xs_curr[1]) < z_min:
            break
    return xs_curr, v_curr

@njit
def gaussian_acc_numba(x, lambda_b, w0_b, g, as_zeta):
    """
    An single-function-call Numba-jitted version of the Gaussian beam acceleration to test speedups.
    x: (2, N) array [rho, zeta]
    returns: (2, N) array [a_rho, a_zeta]
    """
    rho = x[0]
    zeta = x[1]
    b = 1.0 / (1.0 + zeta * zeta)

    I = b * np.exp(-2.0 * b * rho * rho)
    pref = (lambda_b * lambda_b) / (np.pi * np.pi * w0_b * w0_b)
    du_drho = 4.0 * b * rho * I

    du_dzeta = pref * 2.0 * b * zeta * (1.0 - 2.0 * b * rho * rho) * I

    acc_rho = -du_drho
    acc_zeta = -du_dzeta - g / as_zeta

    out = np.empty_like(x)
    out[0] = acc_rho
    out[1] = acc_zeta
    return out

@njit
def verlet_gaussian_numba(x0, v0, dt, steps, lambda_b, w0_b, g, as_zeta):
    """
    Velocity-Verlet integrator JIT-compiled with Numba.
    Uses analytic Gaussian acceleration.

    x0, v0: (2, N)
    returns: xs, vs, ts
    """
    N = x0.shape[1]

    xs = np.zeros((steps + 1, 2, N))
    vs = np.zeros((steps + 1, 2, N))
    ts = np.zeros(steps + 1)

    xs[0] = x0
    vs[0] = v0
    ts[0] = 0.0

    # First step (Taylor)
    a0 = gaussian_acc_numba(xs[0], lambda_b, w0_b, g, as_zeta)
    xs[1] = xs[0] + v0 * dt + 0.5 * a0 * dt * dt
    ts[1] = dt

    for i in range(1, steps):
        t = (i + 1) * dt
        ts[i+1] = t

        z = xs[i, 1]
        active = z > 0.0

        a = gaussian_acc_numba(xs[i], lambda_b, w0_b, g, as_zeta)

        xs[i+1] = xs[i]
        vs[i] = vs[i-1]

        for j in prange(N):
            if active[j]:
                xs[i+1, 0, j] = xs[i, 0, j] + (xs[i, 0, j] - xs[i-1, 0, j] + a[0, j] * dt * dt)
                xs[i+1, 1, j] = xs[i, 1, j] + (xs[i, 1, j] - xs[i-1, 1, j] + a[1, j] * dt * dt)

                vs[i, 0, j] = (xs[i+1, 0, j] - xs[i-1, 0, j]) / (2.0 * dt)
                vs[i, 1, j] = (xs[i+1, 1, j] - xs[i-1, 1, j]) / (2.0 * dt)

    last_z = xs[-1, 1]
    last_active = last_z > 0.0
    vs[-1] = vs[-2]
    for j in prange(N):
        if last_active[j]:
            vs[-1, 0, j] = (xs[-1, 0, j] - xs[-2, 0, j]) / dt
            vs[-1, 1, j] = (xs[-1, 1, j] - xs[-2, 1, j]) / dt

    return xs, vs, ts