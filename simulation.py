from Dynamics import *
from Verlet import *
import numpy as np
import os
from Beams import GaussianBeam, LGBeamL1
from FieldLUT import FieldLUT3D
from tqdm import trange

# MOT characteristics
RMOT = 1e-3 # m
VMOT = 4/3 * np.pi * RMOT**3 # m^3
dMOT_max = 20e-3 # m
h_max = dMOT_max + RMOT # m

T_MAX = 80e-3 # s
N_steps = int(4e3)
DT = T_MAX / N_steps # s

N_save = 30 # number of saved steps
DT_save = DT * N_save # s

GaussBeam_Lambda = 1064e-9 # m
LGBeam_Lambda = 532e-9 # m

# FLAGS
Diff_Powers = True # If True saves data in specific folder for postprocessing


def _make_res_folder(T, dMOT, beam_name, P_b, HEATING, output_beam_name=None):
    if output_beam_name is None:
        output_beam_name = beam_name
    if Diff_Powers:
        return (
            data_folder
            + f"{output_beam_name}/Different_Powers/"
            f"res_T={T*1e6:.0f}uK_dMOT={dMOT*1e3:.0f}mm_P={P_b}W/"
        )
    if HEATING:
        out_folder = output_beam_name + "/Heating"
    else:
        out_folder = output_beam_name
    return data_folder + f"{out_folder}/res_T={T*1e6:.0f}uK_dMOT={dMOT*1e3:.0f}mm/"


def _save_initial_condition_plot(x0, beam, lut_info=None, max_points=20000):
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = False
    x0 = np.asarray(x0)
    if x0.ndim != 2 or x0.shape[0] < 2:
        raise ValueError("x0 must have shape (2, N) for initial condition plot")
    rho = x0[0]
    zeta = x0[1]

    if rho.size > max_points:
        idx = np.random.choice(rho.size, size=max_points, replace=False)
        rho = rho[idx]
        zeta = zeta[idx]

    if lut_info and "field" in lut_info:
        field = lut_info["field"]
        intensity_xz, intensity_yz = field.intensity_planes(field.x, field.y, field.z)
        s_r = lut_info["s_r"]
        s_z = lut_info["s_z"]
        x_axis = field.x / s_r
        y_axis = field.y / s_r
        z_axis = field.z / s_z
    else:
        x_span = float(np.max(rho) - np.min(rho))
        z_span = float(np.max(zeta) - np.min(zeta))
        pad = 0.1
        x_axis = np.linspace(
            float(np.min(rho) - pad * x_span),
            float(np.max(rho) + pad * x_span),
            200,
        )
        z_axis = np.linspace(
            float(np.min(zeta) - pad * z_span),
            float(np.max(zeta) + pad * z_span),
            200,
        )
        X, Z = np.meshgrid(x_axis, z_axis, indexing="ij")
        intensity_xz = beam.intensity(np.abs(X), Z)
        intensity_yz = intensity_xz.copy()
        y_axis = x_axis

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, axis_vals, intensity, label, horiz_label in (
        (axes[0], x_axis, intensity_xz, "xz plane", "x (w0 units)"),
        (axes[1], y_axis, intensity_yz, "yz plane", "y (w0 units)"),
    ):
        mesh = ax.pcolormesh(
            axis_vals,
            z_axis,
            intensity.T,
            shading="auto",
            cmap="viridis",
        )
        ax.scatter(rho, zeta, s=3, alpha=0.25, edgecolors="none", color="white")
        ax.set_xlabel(horiz_label)
        ax.set_ylabel("zeta (zR units)")
        ax.set_title(f"Initial condition ({label})")
        fig.colorbar(mesh, ax=ax, label="Intensity")

    os.makedirs("media", exist_ok=True)
    out_path = os.path.join("media", "initial_condition.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def print_simulation_parameters(
    N, T, dMOT, RMOT,
    w0, zR, tau,
    m_Rb, kB,
    rho_max, zeta_min, zeta_max,
    t_max, dt, N_steps,
    beam_name, P_b, lambda_b, w0_b,
    lut_info=None,
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

    if lut_info is not None:
        print("\n--- LUT ---")
        print(
            "grid: "
            f"{lut_info['grid'][0]} x {lut_info['grid'][1]} x {lut_info['grid'][2]}"
        )
        unit = lut_info.get("axis_unit")
        unit_label = f" {unit}" if unit else ""
        x_min, x_max = lut_info["x_range"]
        y_min, y_max = lut_info["y_range"]
        z_min, z_max = lut_info["z_range"]
        print(f"x range: {x_min:.3g} .. {x_max:.3g}{unit_label}")
        print(f"y range: {y_min:.3g} .. {y_max:.3g}{unit_label}")
        print(f"z range: {z_min:.3g} .. {z_max:.3g}{unit_label}")
        rho_min, rho_max_req = lut_info["rho_range"]
        zeta_min_req, zeta_max_req = lut_info["zeta_range"]
        print(f"required rho: {rho_min:.3g} .. {rho_max_req:.3g}")
        print(f"required zeta: {zeta_min_req:.3g} .. {zeta_max_req:.3g}")
        x_req_min, x_req_max = lut_info["x_req_range"]
        z_req_min, z_req_max = lut_info["z_req_range"]
        print(f"mapped x: {x_req_min:.3g} .. {x_req_max:.3g}{unit_label}")
        print(f"mapped z: {z_req_min:.3g} .. {z_req_max:.3g}{unit_label}")
        print(f"r_ref: {lut_info['r_ref']:.3g} m")
        if "outside_mode" in lut_info:
            print(f"outside mode: {lut_info['outside_mode']}")

    print("\n==============================\n")

def write_params_to_file(
    res_folder: str,
    N: int, T: float, dMOT: float, RMOT: float, # MOT params
    beam_name: str, P_b: float, lambda_b: float, w0_b: float, HEATING: bool, # Beam params
    w0: float, zR: float, tau: float, # length and time scales
    rho_max: float, zeta_min: float, zeta_max: float, # max/min position values
    t_max: float, dt: float, N_steps: int, # time steps
    lut_info=None,
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

        if lut_info is not None:
            f.write("\n--- LUT ---")
            f.write(
                f"\ngrid: {lut_info['grid'][0]} x {lut_info['grid'][1]} x "
                f"{lut_info['grid'][2]}"
            )
            unit = lut_info.get("axis_unit")
            unit_label = f" {unit}" if unit else ""
            x_min, x_max = lut_info["x_range"]
            y_min, y_max = lut_info["y_range"]
            z_min, z_max = lut_info["z_range"]
            f.write(f"\nx range: {x_min:.3g} .. {x_max:.3g}{unit_label}")
            f.write(f"\ny range: {y_min:.3g} .. {y_max:.3g}{unit_label}")
            f.write(f"\nz range: {z_min:.3g} .. {z_max:.3g}{unit_label}")
            rho_min, rho_max_req = lut_info["rho_range"]
            zeta_min_req, zeta_max_req = lut_info["zeta_range"]
            f.write(f"\nrequired rho: {rho_min:.3g} .. {rho_max_req:.3g}")
            f.write(f"\nrequired zeta: {zeta_min_req:.3g} .. {zeta_max_req:.3g}")
            x_req_min, x_req_max = lut_info["x_req_range"]
            z_req_min, z_req_max = lut_info["z_req_range"]
            f.write(f"\nmapped x: {x_req_min:.3g} .. {x_req_max:.3g}{unit_label}")
            f.write(f"\nmapped z: {z_req_min:.3g} .. {z_req_max:.3g}{unit_label}")
            f.write(f"\nr_ref: {lut_info['r_ref']:.3g} m\n")
            if "outside_mode" in lut_info:
                f.write(f"\noutside mode: {lut_info['outside_mode']}\n")

        f.write("\n==============================\n")

def simulation(
    N=int(1e5),
    T=15,
    dMOT=5,
    beam=GaussianBeam(),
    HEATING=False,
    lut_info=None,
    output_beam_name=None,
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

    beam_name = beam.name
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
    save_init_plot = os.environ.get("SAVE_INIT_PLOT", "1").strip()
    save_init_plot = save_init_plot not in ("0", "false", "False")
    if save_init_plot:
        _save_initial_condition_plot(x0, beam, lut_info=lut_info)

    # Call this after defining constants in your script
    if __name__=='__main__':
        print_simulation_parameters(
            N=N, T=T, dMOT=dMOT, RMOT=RMOT, t_max=T_MAX,
            w0=w0, zR=zR, tau=tau, m_Rb=m_Rb, kB=kB,
            rho_max=rho_max, zeta_min=zeta_min, zeta_max=zeta_max,
            dt=DT, N_steps=N_steps,
            beam_name=beam_name, P_b=P_b, lambda_b=beam.lambda_b, w0_b=beam.w0_b,
            lut_info=lut_info,
        )

    # # First stage: same as before (Python evolve_up_to)
    x_prepared, v_prepared = evolve_up_to(
        x0=x0,
        v0=v0,
        acc=beam.acc,
        dt=dt,
        N_steps=N_steps,
        z_min=10,
        beam=beam,
        HEATING=HEATING
    )

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
        
    # Save data and parameters
    save_data(res,
        N, T, dMOT, RMOT, # MOT params
        beam_name, beam.P_b, beam.lambda_b, beam.w0_b, HEATING, # Beam params
        w0, zR, tau, # length and time scales
        rho_max, zeta_min, zeta_max, # max/min position values
        t_max=T_MAX, dt=dt, N_steps=N_steps, # time steps
        lut_info=lut_info,
        output_beam_name=output_beam_name,
    )

def evolve_up_to(x0, v0, acc, dt, N_steps, z_min=5, beam=None, HEATING=False):
    res = verlet_up_to(x0, v0, acc, dt, N_steps, z_min=z_min, beam=beam, HEATING=HEATING)
    return res

def save_data(res, 
    N: int, T: float, dMOT: float, RMOT: float, # MOT params
    beam_name: str, P_b: float, HEATING: bool, # Beam params
    w0: float, zR: float, tau: float, # length and time scales
    rho_max: float, zeta_min: float, zeta_max: float, # max/min position values
    t_max: float, dt: float, N_steps: int, # time steps
    lut_info=None,
    output_beam_name=None,
):
    """
    Save raw simulation results and main parameters to disk.
    """

    res_folder = _make_res_folder(
        T,
        dMOT,
        beam_name,
        P_b,
        HEATING,
        output_beam_name=output_beam_name,
    )

    os.makedirs(res_folder, exist_ok=True)
    print(f"Saving results to {res_folder}")

    # Save arrays
    iterator = trange(0, 3, desc="Saving", mininterval=1.0)

    f_names = [pos_fname, vel_fname, time_fname]

    idx = np.linspace(0, len(res[0])-1, N_save, dtype=int)

    for i in iterator:
        small_res = res[i]
        small_res = small_res[idx]
        np.savez_compressed(res_folder + f_names[i], small_res)
    
    # Save parameters in a human-readable text file
    beam = beams[beam_name]
    write_params_to_file(res_folder,
                         N, T, dMOT, RMOT,
                         beam_name, P_b, beam.lambda_b, beam.w0_b, HEATING,
                         w0, zR, tau,
                         rho_max, zeta_min, zeta_max,
                         t_max, dt, N_steps,
                         lut_info=lut_info)


if __name__ == '__main__':
    from sys import argv

    try:
        if len(argv) < 3:
            print('Specify T and dMOT')
            exit()

        T = int(argv[1])
        dMOT = int(argv[2])
        beam_name = argv[3]
        output_beam_name = beam_name
        use_lut_mode = beam_name == "LUT"
        base_beam_name = beam_name
        if use_lut_mode:
            base_beam_name = os.environ.get("FIELD_LUT_BASE_BEAM", "Gauss")
            if base_beam_name not in ("Gauss", "LG"):
                raise ValueError("FIELD_LUT_BASE_BEAM must be Gauss or LG")
            beam_name = base_beam_name

        if len(argv) > 4 and argv[4] not in ("", "None", "none"):
            P_b = float(argv[4]) # power beam (W)
        else:
            P_b = 1

        if len(argv) > 5 and argv[5] is not None:
            token = str(argv[5]).strip()
            if token in ("1", "True", "true", "TRUE"):
                HEATING = True
            elif token in ("0", "False", "false", "FALSE", ""):
                HEATING = False
            else:
                raise ValueError("HEATING must be True/False or 1/0")
        else:
            HEATING = False

        if beam_name == 'LG':
            beam = LGBeamL1(P_b=P_b, lambda_b=LGBeam_Lambda, w0_b=19e-6)
        elif beam_name == 'Gauss':
            beam = GaussianBeam(P_b=P_b, lambda_b=GaussBeam_Lambda, w0_b=19e-6)
        else:
            raise ValueError(f"Unknown beam name: {beam_name}")

        lut_path = None
        if use_lut_mode:
            if len(argv) > 6 and argv[6] not in ("", "None", "none"):
                lut_path = argv[6]
            if lut_path is None:
                lut_path = os.environ.get("FIELD_LUT_H5", None)
            if not lut_path:
                lut_path = "input/field_data.h5"

        lut_info = None
        if use_lut_mode:
            expected_axis_unit = os.environ.get("FIELD_LUT_AXIS_UNIT", "r_F")
            if expected_axis_unit.lower() in ("none", ""):
                expected_axis_unit = None
            expected_axis_scale_env = os.environ.get("FIELD_LUT_AXIS_SCALE", "1.0")
            if expected_axis_scale_env.lower() in ("none", ""):
                expected_axis_scale = None
            else:
                expected_axis_scale = float(expected_axis_scale_env)
            acc_scale = float(os.environ.get("FIELD_LUT_ACC_SCALE", "1.0"))
            r_ref = float(os.environ.get("FIELD_LUT_R_REF", R_trap))
            auto_scale = os.environ.get("FIELD_LUT_AUTO_SCALE", "0").strip()
            auto_scale = auto_scale not in ("0", "false", "False")
            grid_pad = float(os.environ.get("FIELD_LUT_GRID_PAD", "0.02"))
            outside_mode = os.environ.get("FIELD_LUT_OUTSIDE", "gravity")
            warn_outside = os.environ.get("FIELD_LUT_WARN_OUTSIDE", "1").strip()
            warn_outside = warn_outside not in ("0", "false", "False")

            zR = beam.zR
            z_max = dMOT * 1e-3 + RMOT
            z_min = dMOT * 1e-3 - RMOT
            zeta_max = z_max / zR
            zeta_min = z_min / zR
            rho_max = h_max / zR
            domain_margin = float(os.environ.get("FIELD_LUT_DOMAIN_MARGIN", "0.1"))
            if domain_margin < 0.0:
                raise ValueError("FIELD_LUT_DOMAIN_MARGIN must be >= 0")
            rho_req_max = rho_max * (1.0 + domain_margin)
            zeta_span = zeta_max - zeta_min
            zeta_req_min = zeta_min - domain_margin * zeta_span
            zeta_req_max = zeta_max + domain_margin * zeta_span

            beam.acc, lut_info = FieldLUT3D.acceleration_from_field_data_h5(
                path=lut_path,
                beam=beam,
                r_ref=r_ref,
                expected_axis_unit=expected_axis_unit,
                expected_axis_scale=expected_axis_scale,
                acc_scale=acc_scale,
                required_rho=(-rho_req_max, rho_req_max),
                required_zeta=(zeta_req_min, zeta_req_max),
                auto_scale=auto_scale,
                grid_pad=grid_pad,
                outside_mode=outside_mode,
                warn_outside=warn_outside,
            )
            lut_info["path"] = lut_path
            lut_info["axis_scale"] = expected_axis_scale
            print(f"Using field LUT acceleration from {lut_path}")
            if use_lut_mode:
                print(f"LUT mode enabled (base beam: {base_beam_name})")
        else:
            # --- enable cached analytic intensity for speed (SciPy-backed) ---
            # tune these bounds / resolutions as needed
            beam.enable_intensity_cache(
                rho_max=2.0,
                Nrho=10000,
                zeta_min=0.0,
                zeta_max=2.0,
                Nzeta=10000,
            )
    # exit()
    except Exception as e:
        print(
            "\nUsage: python ./simulation.py <T> <dMOT> <Beam> <P_b> <HEATING> [LUT_H5]\n"
        )
        print("Beam: Gauss, LG, or LUT (LUT uses FIELD_LUT_BASE_BEAM=Gauss/LG)")
        print("HEATING accepts True/False or 1/0 (default: False)\n")
        print("Error:", e)
        exit()

    try:
        simulation(
            N=int(1e5),
            T=T,
            dMOT=dMOT,
            beam=beam,
            HEATING=HEATING,
            lut_info=lut_info,
            output_beam_name=output_beam_name,
        )
    except Exception as e:
        print(e)
