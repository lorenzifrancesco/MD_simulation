from Dynamics import *
from Verlet import *
import numpy as np
import os
import sys
from Beams import GaussianBeam, LGBeamL1

# MOT characteristics
RMOT = 1e-3 # m
VMOT = 4/3 * np.pi * RMOT**3
dMOT_max = 20e-3 # m
h_max = dMOT_max + RMOT

T_MAX = 80e-3
N_steps = int(4e3)
DT = T_MAX / N_steps

N_save = 30 # number of saved steps
DT_save = DT * N_save

GaussBeam_Lambda = 1064e-9 # m
LGBeam_Lambda = 772e-9 # m

# FLAGS
Diff_Powers = False

def print_simulation_parameters(
    N, T, dMOT, RMOT,
    w0, zR, tau,
    m_Rb, kB,
    rho_max, zeta_min, zeta_max,
    t_max, dt, N_steps,
    beam_name, P_b, lambda_b, w0_b
):
    """
    Print the main simulation parameters and derived quantities.
    """

    # Velocity scales
    vs_rho = w0 / tau
    vs_zeta = zR / tau
    alpha = m_Rb / (2 * kB * T)
    v_rms = np.sqrt(np.pi / alpha)

    print("\n=== SIMULATION PARAMETERS ===")
    print("\n--- MOT parameters ---")
    print(f"T: {T*1e6:.2f} uK")
    print(f"N_atoms (N): {N:.2e}")
    print(f"dMOT: {dMOT*1e3:.2f} mm")
    print(f"RMOT: {RMOT*1e3:.2f} mm")

    print("\n--- Beam parameters ---")
    print(f"Beam_name: {beam_name}")
    print(f"Power: {P_b:.2f} W")
    print(f"Lambda_b: {lambda_b*1e9:.3f} nm")
    print(f"w0_b: {w0_b*1e6:.3f} um")

    print("\n--- Initial positions ---")
    print(f"rho_max: {rho_max:.2e} (w0 units) (r_max = {rho_max * w0:.2e} m)")
    print(f"zeta_min: {zeta_min:.2e} (zR units) (z_min = {zeta_min * zR:.2e} m)")
    print(f"zeta_max: {zeta_max:.2e} (zR units) (z_max = {zeta_max * zR:.2e} m)")

    print("\n--- Velocity scales ---")
    print(f"vs_rho: {vs_rho:.2e} m/s")
    print(f"vs_zeta: {vs_zeta:.2e} m/s")
    print(f"alpha: {alpha:.2e} (s/m)**2")
    print(f"v_rms: {v_rms:.2e} m/s")

    print("\n--- Time discretization ---")
    print(f"t_max: {t_max*1e3:.2f} ms")
    print(f"dt: {dt*1e6:.2f} us")
    print(f"N_steps: {N_steps}")

    print("\n==============================\n")

def write_params_to_file(
    res_folder: str,
    N: int, T: float, dMOT: float, RMOT: float, # MOT params
    beam_name: str, P_b: float, lambda_b: float, w0_b: float, HEATING: bool, # Beam params
    w0: float, zR: float, tau: float, # length and time scales
    rho_max: float, zeta_min: float, zeta_max: float, # max/min position values
    t_max: float, dt: float, N_steps: int # time steps
):
    """
    Print the main simulation parameters and derived quantities to parameters.txt.
    """

    # Velocity scales
    vs_rho = w0 / tau
    vs_zeta = zR / tau
    alpha = m_Rb / (2 * kB * T)
    v_rms = np.sqrt(np.pi / alpha)

    param_file = res_folder + "parameters.txt"

    with open(param_file, "w") as f:
        f.write("\n=== SIMULATION PARAMETERS ===")
        f.write("\n--- MOT parameters ---")
        f.write(f"\nT: {T*1e6:.2f} uK")
        f.write(f"\nN_atoms (N): {N:.2e}")
        f.write(f"\ndMOT: {dMOT*1e3:.2f} mm")
        f.write(f"\nRMOT: {RMOT*1e3:.2f} mm\n")

        f.write("\n--- Beam parameters ---")
        f.write(f"\nBeam_name: {beam_name}")
        f.write(f"\nPower: {P_b:.2f} W")
        f.write(f"\nLambda_b: {lambda_b*1e9:.1f} nm")
        f.write(f"\nw0_b: {w0_b*1e6:.1f} um")
        f.write(f"\nHeating: {HEATING} \n")

        f.write("\n--- Initial positions ---")
        f.write(f"\nrho_max: {rho_max:.2e} (w0 units) (r_max = {rho_max * w0:.2e} m)")
        f.write(f"\nzeta_min: {zeta_min:.2e} (zR units) (z_min = {zeta_min * zR:.2e} m)")
        f.write(f"\nzeta_max: {zeta_max:.2e} (zR units) (z_max = {zeta_max * zR:.2e} m)")

        f.write("\n--- Scales ---")
        f.write(f"\nw0: {w0:.3e} m")
        f.write(f"\nzR: {zR:.3e} m")
        f.write(f"\ntau: {tau:.3e} s")
        f.write(f"\nvs_rho: {vs_rho:.2e} m/s")
        f.write(f"\nvs_zeta: {vs_zeta:.2e} m/s")
        f.write(f"\nalpha: {alpha:.2e} (s/m)**2")
        f.write(f"\nv_rms: {v_rms:.2e} m/s")

        f.write("\n--- Time discretization ---")
        f.write(f"\nt_max: {t_max*1e3:.2f} ms")
        f.write(f"\ndt: {dt*1e6:.2f} us")
        f.write(f"\nN_steps: {N_steps}\n")

        f.write("\n==============================\n")

def simulation(
    N=int(1e5),
    T=15,
    dMOT=5,
    beam=GaussianBeam(),
    HEATING=False,
):
    
    """
    Run a full atom trajectory simulation.

    Parameters
    ----------
    N : int, optional
        Number of atoms (default: 1e2 for test, 1e5 recommended).
    T : float
        Temperature [µK].
    dMOT : float
        MOT–fiber distance [mm].

    Returns
    -------
    None
        Runs the integration, saves results in `./data/`.

    Notes
    -----
    - Positions initialized uniformly within MOT sphere.
    - Velocities drawn from thermal Maxwell-Boltzmann distribution.
    - Calls `verlet` integrator from `Dynamics.py`.
    - Results are saved with filenames based on T and dMOT.
    """

    # SIMULATION PARAMETERS
    N = int(N) # num of atoms

    # MOT
    T = T * 1e-6 # K
    dMOT = dMOT * 1e-3 # m

    zR = beam.zR
    vs_rho = beam.vs_rho
    vs_zeta = beam.vs_zeta
    tau = beam.tau
    w0 = beam.w0_b

    # initial positions
    z_max = dMOT + RMOT
    z_min = dMOT - RMOT
    zeta_max = z_max / zR
    zeta_min = z_min / zR

    zeta_0 = np.random.uniform(zeta_min, zeta_max, size=N)

    rho_max = h_max / zR # in units of w0
    rho_0 = np.random.uniform(-rho_max, rho_max, size=N)

    x0 = np.array([rho_0, zeta_0])

    # initial velocities
    alpha = m_Rb / (2 * kB * T)
    sigma_rho = np.sqrt(1 / (2*alpha)) / vs_rho
    sigma_zeta = np.sqrt(1 / (2*alpha)) / vs_zeta
    v_rho_0 = np.random.normal(loc = 0, scale = sigma_rho, size = N)
    v_zeta_0 = np.random.normal(loc = 0, scale = sigma_zeta, size = N)

    v0 = np.array([v_rho_0, v_zeta_0])

    # Time and Num
    dt = DT / tau

    # Call this after defining constants in your script
    if __name__=='__main__':
        print_simulation_parameters(
            N=N, T=T, dMOT=dMOT, RMOT=RMOT, t_max=T_MAX,
            w0=w0, zR=zR, tau=tau, m_Rb=m_Rb, kB=kB,
            rho_max=rho_max, zeta_min=zeta_min, zeta_max=zeta_max,
            dt=DT, N_steps=N_steps,
            beam_name=beam_name, P_b=P_b, lambda_b=beam.lambda_b, w0_b=beam.w0_b
        )

    # # First stage: same as before (Python evolve_up_to)
    x_prepared, v_prepared = evolve_up_to(
        x0=x0,
        v0=v0,
        acc=beam.acc,
        dt=dt,
        N_steps=N_steps,
        z_min=10,
        HEATING=HEATING
    )

    if True:
        res = verlet(
            x0=x_prepared,
            v0=v_prepared,
            a_func=beam.acc,
            dt=dt,
            N_steps=N_steps,
            N_saves=N_save, # new param!
            beam=beam,
            HEATING=HEATING
        )
    else:
        res = verlet_gaussian_numba(
            x0=x_prepared,
            v0=v_prepared,
            dt=dt,
            N_steps=N_steps,
            N_saves=N_save,  # new param!
            beam=beam,
            HEATING=HEATING,
        )

        
    # Save data and parameters
    save_data(res, 
        N, T, dMOT, RMOT, # MOT params
        beam_name, beam.P_b, beam.lambda_b, beam.w0_b, HEATING, # Beam params
        w0, zR, tau, # length and time scales
        rho_max, zeta_min, zeta_max, # max/min position values
        t_max=T_MAX, dt=dt, N_steps=N_steps # time steps
    )

def evolve_up_to(x0, v0, acc, dt, N_steps, z_min=5, HEATING=False):
    res = verlet_up_to(x0, v0, acc, dt, N_steps, z_min=z_min, HEATING=HEATING)
    return res

def save_data(res, 
    N: int, T: float, dMOT: float, RMOT: float, # MOT params
    beam_name: str, P_b: float, lambda_b: float, w0_b: float, HEATING: bool, # Beam params
    w0: float, zR: float, tau: float, # length and time scales
    rho_max: float, zeta_min: float, zeta_max: float, # max/min position values
    t_max: float, dt: float, N_steps: int # time steps
):
    """
    Save raw simulation results and main parameters to disk.
    """

    if Diff_Powers:
        res_folder = data_folder + f'{beam_name}/Different_Powers/res_T={T*1e6:.0f}uK_dMOT={dMOT*1e3:.0f}mm_P={P_b}W/'
    else:
        if HEATING:
            out_folder = beam_name + '/Heating'
        else:
            out_folder = beam_name
            
        res_folder = data_folder + f'{out_folder}/res_T={T*1e6:.0f}uK_dMOT={dMOT*1e3:.0f}mm/'

    os.makedirs(res_folder, exist_ok=True)

    # Save arrays
    iterator = trange(0, 3, desc="Saving", mininterval=1.0)

    f_names = [pos_fname, vel_fname, time_fname]

    idx = np.linspace(0, len(res[0])-1, N_save, dtype=int)

    for i in iterator:
        small_res = res[i]
        small_res = small_res[idx]
        np.save(res_folder + f_names[i], small_res)

    # Save parameters in a human-readable text file
    write_params_to_file(res_folder,
                         N, T, dMOT, RMOT,
                         beam_name, P_b, beam.lambda_b, beam.w0_b, HEATING,
                         w0, zR, tau,
                         rho_max, zeta_min, zeta_max,
                         t_max, dt, N_steps)


if __name__ == '__main__':
    from sys import argv

    try:
        if len(argv) < 3:
            print('Specify T and dMOT')
            exit()

        T = int(argv[1])
        dMOT = int(argv[2])
        beam_name = argv[3]

        if argv[4] != None:
            P_b = float(argv[4]) # power beam (W)
        else:
            P_b = 1

        if argv[5] != None:
            HEATING = bool(argv[5])
        else:
            HEATING = False

        if beam_name == 'LG':
            beam = LGBeamL1(P_b=P_b, lambda_b=LGBeam_Lambda, w0_b=19e-6)
        elif beam_name == 'Gauss':
            beam = GaussianBeam(P_b=P_b, lambda_b=GaussBeam_Lambda, w0_b=19e-6)
        else:
            raise ValueError(f"Unknown beam name: {beam_name}")

        # --- enable LUT-based intensity for speed (SciPy-backed) ---
        # tune these bounds / resolutions as needed
        beam.enable_intensity_lut(
            rho_max=2.0,    # dimensionless rho range you care about
            Nrho=100,
            zeta_min=0.0,   # use negative if particles explore zeta < 0
            zeta_max=2.0,
            Nzeta=100,
        ) # with this LUT complexity, in 3D we would obtain a 500x500x500 grid
        beam.disable_intensity_lut()
        
    # exit()
    except Exception as e:
        print("\nUsage: python ./simulation.py <T> <dMOT> <Beam> <P_b> <HEATING>\n")
        print("Error:", e)
        exit()

    try:
        simulation(N=int(1e5), T=T, dMOT=dMOT, beam=beam, HEATING=HEATING)
    except Exception as e:
        print(e)