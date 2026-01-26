from simulation import *
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib as mpl
import os
import re
import h5py
from scipy.ndimage import map_coordinates
from Beams import beams
from Heating import GetTemperature
from GifsMaker import MakeGif_density

mpl.rcParams["text.usetex"] = False

USE_LUT_INTENSITY = False
LUT_H5_PATH = "input/field_data.h5"
LUT_VERBOSE = False
PLOT_MODE = "save"
PLOT_DIR = "media"


def _finalize_plot(name):
    mode = PLOT_MODE
    if mode not in ("save", "show", "both"):
        raise ValueError("PLOT_MODE must be 'save', 'show', or 'both'")
    if mode in ("save", "both"):
        os.makedirs(PLOT_DIR, exist_ok=True)
        out_path = os.path.join(PLOT_DIR, f"{name}.png")
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot to {out_path}")
    if mode in ("show", "both"):
        plt.show()
    plt.close()

def data_fname(T, dMOT, beam_name, middle_folder=''):
    res_fname = f'res_T={T:.0f}uK_dMOT={dMOT:.0f}mm/'
    if middle_folder == '':
        simul_path = data_folder + f'{beam_name}/{res_fname}'
    else:
        simul_path = data_folder + f'{beam_name}/{middle_folder}/{res_fname}/'
    if os.path.exists(simul_path):
        return simul_path
    print(f'No simulation present at {simul_path}')


def _lut_intensity_grid(rho_array, zeta_array, beam, simul_path):
    r_ref = GetParam(simul_path, param="r_ref")
    if r_ref is None:
        raise ValueError("r_ref not found in parameters.txt")

    s_r = beam.w0_b / r_ref
    s_z = beam.zR / r_ref

    rho_array = np.asarray(rho_array, dtype=float)
    zeta_array = np.asarray(zeta_array, dtype=float)
    x_field = np.abs(rho_array) * s_r
    z_field = zeta_array * s_z

    with h5py.File(LUT_H5_PATH, "r") as f:
        axis_scale = float(f.attrs.get("axis_scale", 1.0))
        if LUT_VERBOSE and not getattr(_lut_intensity_grid, "_logged", False):
            _lut_intensity_grid._logged = True
            print(
                "LUT intensity enabled:",
                f"path={LUT_H5_PATH}, axis_scale={axis_scale}, r_ref={r_ref:.3g}",
            )
        z_axis = np.asarray(f["domain/z"][:], dtype=float) * axis_scale
        fields_group = f["fields"]

        intensity_xz = np.empty((z_axis.size, x_field.size), dtype=float)
        for idx in range(z_axis.size):
            g = fields_group[f"z_{idx:05d}"]
            x = np.asarray(g["x"][:], dtype=float) * axis_scale
            y = np.asarray(g["y"][:], dtype=float) * axis_scale
            intensity = np.asarray(g["intensity"][:], dtype=float)

            dx = x[1] - x[0]
            dy = y[1] - y[0]
            ix = (x_field - x[0]) / dx
            iy0 = (0.0 - y[0]) / dy
            coords = np.vstack([ix, np.full_like(ix, iy0)])
            intensity_xz[idx, :] = map_coordinates(
                intensity,
                coords,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )

    intensity = np.empty((z_field.size, x_field.size), dtype=float)
    for j in range(x_field.size):
        intensity[:, j] = np.interp(
            z_field, z_axis, intensity_xz[:, j], left=0.0, right=0.0
        )
    return intensity


def LoadTime(simul_path: str):
    """
    Load simulation time array from the folder.
    
    Parameters
    ----------
    simul_path : str
        Path to the simulation folder.

    Returns
    -------
    ts : np.ndarray
        Time points array.
    """
    try:
        ts = np.load(simul_path + time_fname, allow_pickle=True)
        return ts
    except FileNotFoundError:
        print(f"Time file not found in {simul_path}")
        return None

def LoadPosition(simul_path: str):
    """
    Load particle positions from the simulation folder.

    Returns
    -------
    xs : np.ndarray
        Shape: (time_steps, 2, N_atoms) -> rho, zeta
    """
    try:
        xs = np.load(simul_path + pos_fname, allow_pickle=True)
        return xs
    except FileNotFoundError:
        print(f"Position file not found in {simul_path}")
        return None

def LoadVelocity(simul_path: str):
    """
    Load particle velocities from the simulation folder.

    Returns
    -------
    vs : np.ndarray
        Shape: (time_steps, 2, N_atoms) -> vrho, vzeta
    """
    try:
        vs = np.load(simul_path + vel_fname, allow_pickle=True)
        return vs
    except FileNotFoundError:
        print(f"Velocity file not found in {simul_path}")
        return None

def get_file_content(file_path):
    """
    Reads the content of the simulation parameter file.
    """
    try:
        # 'r' mode means read-only
        with open(file_path, 'r') as file:
            # .read() reads the entire file content into a single string
            file_content = file.read()
            return file_content
            
    except FileNotFoundError:
        print(f"Error: Parameter file not found at {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return None
    
def GetParam(simul_path: str, param: str):
    """
    Reads a numerical parameter from the provided text content.
    
    Args:
        simul_path (str): Path to simulation folder.
        param (str): The name of the parameter to retrieve (e.g., 'Temperature', 'dMOT', 'm_Rb').
    
    Returns:
        float or None: The numerical value of the parameter, or None if not found.
    """
    # Escape any special characters in the parameter name (though unlikely needed here)
    escaped_param = re.escape(param) 
    
    pattern = rf'({escaped_param}).*?:?\s*([+-]?\d+\.?\d*(?:[Ee][+-]?\d+)?)'

    if simul_path is None:
        simul_path = data_fname(T, dMOT, beam_name)
    file_content = get_file_content(simul_path + 'parameters.txt')

    match = re.search(pattern, file_content, re.IGNORECASE)

    if match:
        try:
            # The number is captured in group 2
            return float(match.group(2))
        except ValueError:
            print(f"Error: Value '{match.group(2)}' for parameter '{param}' is not a valid number.")
            return None
    else:
        # Check for multi-word parameters by searching line-by-line using simpler logic
        # This handles keys like 'MOT displacement (dMOT)'
        for line in file_content.splitlines():
            if param in line:
                 # Strip units/trailing text and try to extract the last 'word'
                 parts = line.split(':')
                 if len(parts) > 1:
                     value_part = parts[1].split()[0] # Take the first word after the colon
                     try:
                         return float(value_part)
                     except ValueError:
                         # Value might be a string (like 'LG'), which is okay for this check
                         return value_part
        
        print(f'Parameter "{param}" not found.')
        return None

def GetInitNumAtoms(simul_path: str):
    N = GetParam(simul_path, param='N_atoms (N)')
    return N

def Get_Power(simul_path: str):
    P = GetParam(simul_path, param='Power') # W
    return P

def Get_Lambda(simul_path: str):
    lambda_b = GetParam(simul_path, param='Lambda_b') # nm
    return lambda_b*1e-9

def Get_Waist(simul_path: str):
    lambda_b = GetParam(simul_path, param='w0_b') # um
    return lambda_b*1e-6

def Get_Beam_Name(simul_path: str):
    name = GetParam(simul_path, param='Beam_name')
    return name

def Get_Beam(simul_path: str):

    beam_name = Get_Beam_Name(simul_path)
    P_b = Get_Power(simul_path)
    Lambda_b = Get_Lambda(simul_path)
    w0_b = Get_Waist(simul_path)

    beam = beams[beam_name]
    beam.Set_Power(P_b)
    beam.Set_Lambda(Lambda_b)
    beam.Set_w0(w0_b)
    beam.update_props()

    beam.Set_Power(P_b)
    beam.update_props()

    return beam

def compute_NMOT(N, R_cil):
    """
    Compute the effective number of atoms in the cylindrical fiber volume.

    Parameters
    ----------
    N : int
        Total number of atoms in the MOT.
    h_cil : float
        Height of the cilynder
    R_cil : float
        Radius of the cilynder

    Returns
    -------
    NMOT : float
        Effective number of atoms inside the cylindrical fiber volume.

    Notes
    -----
    - Uses global constants:
        * VMOT : volume of the MOT sphere
    - Formula:
        NMOT = N * VMOT / V_cil
    """
    
    V_cil = 2 * RMOT * np.pi*R_cil**2

    return N * VMOT / V_cil

def get_frac(simul_path: str, steps: np.ndarray):
    """
    Compute fraction of captured atoms vs at time step.

    Parameters
    ----------
    simul_path: str
        Path to simulation folder.
    steps: np.ndarray
        Numpy array of the steps for which atom fraction is needed.

    Returns
    -------
    f_cap : ndarray
        Fraction of captured atoms at step.

    Notes
    -----
    An atom is considered captured if:
    - ζ <= 0 (at or below fiber tip),
    - |ρ| < R_trap / w0 (within fiber mode radius).
    """

    xs = np.load(simul_path + pos_fname, allow_pickle=True)
    r_cap = R_trap / w0 # trap radius in units of w0
    n_cap = np.sum((xs[steps, 1, :] <= 0) & (np.abs(xs[steps, 0, :]) < r_cap), axis=1)

    N = GetInitNumAtoms(simul_path)
    R_cil = np.max(xs[0, 0, :]) * w0
    NMOT = compute_NMOT(N, R_cil)

    f_cap = n_cap / NMOT
    if len(f_cap) > 1:
        return f_cap
    else:
        return f_cap[0]

def capt_frac_vs_t(simul_path: str):
    """
    Compute fraction of captured atoms vs time.

    Parameters
    ----------
    simul_path: str
        Path to simulation folder.

    Returns
    -------
    ts : ndarray
        Time points (dimensionless).
    f_cap : ndarray
        Fraction of captured atoms at each time.

    Notes
    -----
    An atom is considered captured if:
    - ζ <= 0 (at or below fiber tip),
    - |ρ| < R_trap / w0 (within fiber mode radius).
    """

    ts = np.load(simul_path + time_fname, allow_pickle=True)
    steps = np.arange(0, len(ts), 1, dtype=int)

    f_cap = get_frac(simul_path, steps)

    return ts, f_cap

def density_at_fib(simul_path, step: int):
    """
    Compute radial density distribution of atoms at the fiber.

    Parameters
    ----------
    simul_path: str
        Path to simulation folder.
    step : int
        Time step index (-1 for final distribution).

    Returns
    -------
    hist_rho_step : tuple
        Histogram (counts, bins) for atoms at fiber at given step.
    hist_rho_init : tuple
        Histogram (counts, bins) for initial MOT distribution.
    """

    xs = np.load(simul_path + pos_fname, allow_pickle=True)
    
    rho_step = xs[step, 0, :]
    zeta_step = xs[step, 1, :]
    index_at_fib = zeta_step <= 0

    rho_at_fib = rho_step[index_at_fib]
    N_at_fib = len(rho_at_fib)

    rho_init = xs[0, 0, :]
    N_init = len(rho_init)

    rho_max = 15
    
    rho_init = rho_init[(np.abs(rho_init) < rho_max)]
    rho_at_fib = rho_at_fib[(np.abs(rho_at_fib) < rho_max)]

    if len(rho_at_fib) > 0:
        hist_rho_step = np.histogram(rho_at_fib, int(np.sqrt(N_at_fib)), density=True)
        hist_rho_init = np.histogram(rho_init, int(np.sqrt(N_init)), density=True)

        return hist_rho_step, hist_rho_init
    else:
        return None, None


def z_density(simul_path, step: int):
    """
    Compute axial density distribution of atoms at a given step.

    Parameters
    ----------
    simul_path: str
        Path to simulation folder.
    step : int
        Time step index (-1 for final distribution).

    Returns
    -------
    hist_zeta_step : tuple
        Histogram (counts, bins) for atomic axial positions at given step.
    hist_zeta_init : tuple
        Histogram (counts, bins) for initial MOT distribution.
    """
    xs = np.load(simul_path + pos_fname, allow_pickle=True)
    
    zeta_step = xs[step, 1, :]
    zeta_init = xs[0, 1, :]

    hist_zeta_step = np.histogram(zeta_step, int(np.sqrt(len(zeta_step))), density=True)
    hist_zeta_init = np.histogram(zeta_init, int(np.sqrt(len(zeta_init))), density=True)
    
    return hist_zeta_step, hist_zeta_init

def density(simul_path, rho_min: float, rho_max: float, 
            zeta_min: float, zeta_max: float, step=-1):


    """
    Compute the 2D spatial density histogram of atoms from simulation data.

    Parameters
    ----------
    simul_path: str
        Path to simulation folder.
    rho_min : float
        Minimum value of the radial coordinate (rho) for the histogram.
    rho_max : float
        Maximum value of the radial coordinate (rho) for the histogram.
    zeta_min : float
        Minimum value of the axial coordinate (zeta) for the histogram.
    zeta_max : float
        Maximum value of the axial coordinate (zeta) for the histogram.
    step : int, optional
        Time step index to use from the simulation data (default is -1, the last step).

    Returns
    -------
    n : ndarray
        2D array of histogram counts for each (rho, zeta) bin. Shape is (len(zeta_centers), len(rho_centers)).
    rho_centers : ndarray
        1D array of bin center positions along the rho axis.
    zeta_centers : ndarray
        1D array of bin center positions along the zeta axis.

    Notes
    -----
    The function expects simulation results to be saved in a folder
    with the naming convention: 'res_T={T}uK_dMOT={dMOT}mm/'.
    If the folder or file does not exist, the function exits.
    """

    xs = np.load(simul_path + pos_fname, allow_pickle=True)
    rho_atoms = xs[step, 0, :]
    zeta_atoms = xs[step, 1, :]

    # Define bin edges
    rho_array = np.linspace(rho_min, rho_max, 101)  # 100 bins
    zeta_array = np.linspace(zeta_min, zeta_max, 101)

    # Compute 2D histogram (counts in each bin)
    n, rho_edges, zeta_edges = np.histogram2d(
        rho_atoms, zeta_atoms, bins=[rho_array, zeta_array]
    )

    # For plotting, use bin centers instead of edges
    rho_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    zeta_centers = 0.5 * (zeta_edges[:-1] + zeta_edges[1:])

    return n.T, rho_centers, zeta_centers

def GetTemp_arrays(simul_path):

    beam_simul = Get_Beam(simul_path)

    ts = np.load(simul_path + time_fname, allow_pickle=True)
    xs = np.load(simul_path + pos_fname, allow_pickle=True)
    vs = np.load(simul_path + vel_fname, allow_pickle=True)
    
    rho = xs[:, 0, :]
    zeta = xs[:, 1, :]
    
    vs_rho = vs[:, 0, :]
    vs_zeta = vs[:, 1, :]

    valid_times = []
    T_rho_list = []
    T_zeta_list = []

    for i in range(len(rho)):
        if np.sum(zeta[i] > 0) == len(zeta[0]):
            in_trap_index = (np.abs(rho[i]) < beam_simul.w(zeta[i]))

            vs_rho_i = vs_rho[i]
            vs_zeta_i = vs_zeta[i]

            vs_in_trap_rho = vs_rho_i[in_trap_index]
            vs_in_trap_zeta = vs_zeta_i[in_trap_index]

            vs_in_trap = np.array([vs_in_trap_rho, vs_in_trap_zeta])
            T_rho, T_zeta = GetTemperature(vs_in_trap)
            T_rho_list.append(T_rho)
            T_zeta_list.append(T_zeta)
            valid_times.append(ts[i])

    return np.array(valid_times), np.array(T_rho_list), np.array(T_zeta_list)
    

def plot_cap_frac(simul_path: str, label='Fraction Captured', color='royalblue'):
    """
    Plot fraction of captured atoms vs time.

    Parameters
    ----------
    simul_path: str
        Path to simulation folder.
    label : str
        Plot label.
    color : str
        Curve color.
    """

    beam = Get_Beam(simul_path)
    
    ts = LoadTime(simul_path)
    steps = np.arange(0, len(ts), 1)

    f_cap = get_frac(simul_path, steps)
    plt.plot(ts * beam.tau * 1e3, f_cap*100, label=label, color=color)
    plt.title('Fraction of atoms captured at the fiber')
    plt.xlabel(r'Time (ms)')
    plt.ylabel('Atoms captured (%)')

def plot_initial_density_rho(hist_rho_init):
    """
    Plot initial radial distribution of MOT atoms.

    Parameters
    ----------
    hist_rho_init : tuple
        Histogram of initial ρ distribution (counts, bins).
    """

    # Bin centers
    init_bins = hist_rho_init[1]

    init_widths = np.diff(init_bins)

    plt.bar(
        init_bins[:-1], hist_rho_init[0],
        width=init_widths, align='edge',
        color='blue', alpha=0.8, label='Initial distribution'
    )

    plt.title('Initial radial distribution of atomic positions')
    plt.xlabel(r'$\rho$ $(w_0)$')
    plt.ylabel('Probability density')
    plt.legend()

def plot_density_at_fib(hist_rho_step, label='Distribution at the fiber', color='red'):
    """
    Plot radial distribution of captured atoms at fiber.

    Parameters
    ----------
    hist_rho_step : tuple
        Histogram of ρ for atoms at fiber.
    label : str
        Plot label.
    color : str
        Bar color.
    """
    # Bin centers
    if hist_rho_step:
        step_bins = hist_rho_step[1]

        step_widths = np.diff(step_bins)

        plt.bar(
            step_bins[:-1], hist_rho_step[0],
            width=step_widths, align='edge',
            color=color, alpha=0.8, label=label
        )
    else:
        plt.plot([],[])

    plt.title('Distribution of radial position at fiber')
    plt.xlabel(r'$\rho$ $(w_0)$')
    plt.ylabel('Probability density')
    plt.xlim(-15, 15)
    plt.legend()

def plot_initial_density_zeta(simul_path: str, hist_zeta_init):

    beam = Get_Beam(simul_path)

    counts, bin_edges = hist_zeta_init

    bin_edges_mm = bin_edges * beam.zR * 1e3
    bin_centers_mm = (bin_edges_mm[:-1] + bin_edges_mm[1:]) / 2
    bin_widths_mm = np.diff(bin_edges_mm)

    plt.bar(
        bin_centers_mm, counts,
        width=bin_widths_mm, align='center',
        color='blue', alpha=0.8, label='Initial distribution'
    )

    plt.title('Initial axial distribution of atomic positions')
    plt.xlabel(r'$z$ $(mm)$')
    plt.ylabel('Probability density')
    plt.legend()


def plot_density_zeta(simul_path: str, hist_zeta_step, label='Distribution of axial positions', color='red'):

    beam = Get_Beam(simul_path)

    counts, bin_edges = hist_zeta_step

    bin_edges_mm = bin_edges * beam.zR * 1e3
    bin_centers_mm = (bin_edges_mm[:-1] + bin_edges_mm[1:]) / 2
    bin_widths_mm = np.diff(bin_edges_mm)

    plt.bar(
        bin_centers_mm, counts,
        width=bin_widths_mm, align='center',
        color=color, alpha=0.8, label=label
    )

    plt.xlabel(r'$z$ $(mm)$')
    plt.ylabel('Probability density')
    plt.legend()


def plot_density_zeta_vs_t(simul_path: str):
    
    ts = LoadTime(simul_path)
    Nt = len(ts)
    steps = np.linspace(0, Nt//2, 5, dtype=int)

    beam = Get_Beam(simul_path)

    cmap = colormaps.get_cmap('inferno')
    colors = [cmap(x) for x in np.linspace(0.1, 0.8, len(steps))]  # <- here

    # Compute initial distribution
    _, hist_zeta_init = z_density(simul_path, steps[0])
    z_max = np.max(hist_zeta_init[1] * beam.zR * 1e3)
    print(f"Max z (mm): {z_max:.3f}")

    for i, step in enumerate(steps):
        hist_zeta_step, _ = z_density(simul_path, step)
        plot_density_zeta(
            simul_path,
            hist_zeta_step,
            label=f't = {step * DT_save * 1e3:.2f} ms',
            color=colors[i]
        )

    plt.title('Axial position distribution at different times')


def plot_density_rho_vs_t(simul_path: str, steps: list):
    """
    Plot axial distribution at given steps.

    Parameters
    ----------
    steps : list
        Steps at which computing the distributions.
    T : float
        MOT temperature [µK].
    dMOT : float
        MOT–fiber distance [mm].
    """

    beam = Get_Beam(simul_path)

    cmap = colormaps.get_cmap('inferno')
    colors = [cmap(x) for x in np.linspace(0.1, 0.8, len(steps))]

    for i, step in enumerate(steps):
        hist_rho_step, _ = density_at_fib(simul_path, step)
        plot_density_at_fib(hist_rho_step, label=f'step = {step}', color=colors[i])

    plt.title('Distribution of atoms at fiber at different times')
    plt.xlim(-200*w0, 200*w0)

def plot_density(simul_path: str, n, rho_array, zeta_array):
    
    """
    Plot a 2D density contour of atomic distribution.

    Parameters
    ----------
    n : ndarray
        2D array of histogram counts (output of `density` function).
    rho_array : ndarray
        1D array of radial bin centers.
    zeta_array : ndarray
        1D array of axial bin centers.

    Returns
    -------
    None
        The function displays a contour plot of the density using matplotlib.

    Notes
    -----
    The radial (r) and axial (z) coordinates are converted to millimeters
    using `w0` and `zR` respectively. The density is plotted using a
    filled contour plot with 50 levels and a 'viridis' colormap.
    """
    
    beam = Get_Beam(simul_path)

    # atomic density contour
    R, Z = np.meshgrid(rho_array * beam.w0_b * 1e3, zeta_array * beam.zR * 1e3)

    fig, ax = plt.subplots(figsize=(8,6))

    # density background
    cp = ax.contourf(R, Z, n, levels=50, cmap="viridis")
    fig.colorbar(cp, ax=ax, label="Atomic Density")

    # beam intensity (normalized)
    if USE_LUT_INTENSITY:
        I = _lut_intensity_grid(rho_array, zeta_array, beam, simul_path)
    else:
        rho_dim = R / (beam.w0_b * 1e3)
        zeta_dim = Z / (beam.zR * 1e3)
        I = beam.intensity(rho_dim, zeta_dim)
    I = I / I.max()

    # overlay with alpha
    cmap = plt.cm.inferno
    cf = ax.contourf(R, Z, I, levels=50, cmap=cmap, alpha=0.1)

    ax.set_title(f'Atom and Intesity distribution ({beam_name})')
    ax.set_xlabel(r'$\rho$ (mm)')
    ax.set_ylabel('z (mm)')

    # make a mappable for the colorbar with opaque colors
    sm = mpl.cm.ScalarMappable(norm=cf.norm, cmap=cmap)
    sm.set_array([])  
    fig.colorbar(sm, ax=ax, label="Beam intensity")

def plot_capfrac_vs_P(beam_name: str):

    folder_path = f"Results/{beam_name}/Different_Powers"

    files = os.listdir(folder_path)

    powers = []
    cp_fracs = []
    for file in files:
    
        match = re.search(r"\d+\.\d+", file)
        if match:
            pw = float(match.group())
            powers.append(pw)
            cp_fracs.append(get_frac(T=15, dMOT=7, beam=beam, step=-1, middle_folder="Different_Powers", fname=file)*100)

    if beam_name == 'Gauss':
        wl = '1064 nm'
    elif beam_name == 'LG':
        wl = '650 nm'
    plt.semilogx(powers, cp_fracs, '--o', label=beam_name + f' {wl}') 

def plot_temperature(simul_path: str):

    ts, T_rho, T_zeta = GetTemp_arrays(simul_path)
    beam = Get_Beam(simul_path)
    
    plt.semilogy(ts * beam.tau * 1e3, T_rho * 1e6, 'o--', label='Radial Temp.')
    plt.semilogy(ts * beam.tau * 1e3, T_zeta * 1e6, 'o--', label='Axial Temp.')
    plt.xlabel('Time (ms)')
    plt.ylabel(r'Temperature ($\mu K$)')
    plt.title('Temperature vs Time')
    plt.grid()
    plt.legend()

def CreateGif_desnity(simul_path: str, beam: Beam, beam_label=None):

    xs= LoadPosition(simul_path)
    z_max = np.max(xs[:, 1, :])
    n_list = []
    rho_list = []
    zeta_list = []

    for i in range(len(xs)):
        n, rho_array, zeta_array = density(simul_path, rho_min=-1.5*RMOT/w0, rho_max=1.5*RMOT/w0, zeta_min=0, zeta_max=z_max, step=i)
        n_list.append(n)
        rho_list.append(rho_array)
        zeta_list.append(zeta_array)

    rho_array = np.array(rho_list)
    zeta_array = np.array(zeta_list)
    n_array = np.array(n_list)

    rho_base = rho_array[0]
    zeta_base = zeta_array[0]
    intensity_grid = None
    if USE_LUT_INTENSITY:
        intensity_grid = _lut_intensity_grid(rho_base, zeta_base, beam, simul_path)

    label = beam_label if beam_label is not None else beam.name
    print(f'Creating GIF for {label}')
    print('rho_array: ', rho_array.shape)
    print('zeta_array: ', zeta_array.shape)
    print('n_array: ', n_array.shape)

    MakeGif_density(
        pos=np.array([rho_base, zeta_base]),
        density=n_array,
        beam=beam,
        file_name=f'density_gif_Beam={label}',
        intensity_grid=intensity_grid,
        beam_label=label,
    )

if __name__ == '__main__':

    from sys import argv

    if len(argv) < 3:
        print('Specify T, dMOT, Beam (Gauss, LG, or LUT)')
        exit()

    try:
        T = int(argv[1])
        dMOT = int(argv[2])
        beam_name = str(argv[3])
        if len(argv) > 4:
            token = str(argv[4]).strip().lower()
            if token in ("1", "true"):
                Heating = True
            elif token in ("0", "false", ""):
                Heating = False
            else:
                raise ValueError("Heating must be True/False or 1/0")
        else:
            Heating = False

        print(f'T = {T} uK, dMOT = {dMOT} mm, beam = {beam_name}, Heating = {Heating}\n')

        plot_mode = os.environ.get("PLOT_MODE", PLOT_MODE).strip().lower()
        plot_dir = os.environ.get("PLOT_DIR", PLOT_DIR).strip()
        if plot_dir:
            PLOT_DIR = plot_dir
        PLOT_MODE = plot_mode if plot_mode else PLOT_MODE

        if beam_name == "LUT":
            USE_LUT_INTENSITY = True
            LUT_VERBOSE = True
            LUT_H5_PATH = os.environ.get("FIELD_LUT_H5", LUT_H5_PATH)
            if not os.path.exists(LUT_H5_PATH):
                raise FileNotFoundError(f"LUT file not found: {LUT_H5_PATH}")
            print(f"Analysis using LUT intensity from {LUT_H5_PATH}")

        simul_path = data_fname(T, dMOT, beam_name)
        if Heating:
            simul_path = data_fname(T, dMOT, beam_name, 'Heating')
        if simul_path is None:
            raise FileNotFoundError("Simulation folder not found; check inputs.")

        plot_cap_frac(simul_path)
        _finalize_plot("cap_frac")

        hist_rho_step, hist_rho_init = density_at_fib(simul_path, step=-1)
        plot_initial_density_rho(hist_rho_init)
        plot_density_at_fib(hist_rho_step=hist_rho_step)
        _finalize_plot("rho_density")

        plot_density_zeta_vs_t(simul_path)
        plt.ylim(0, 1)
        _finalize_plot("zeta_vs_t")

        steps=np.array([-1])
        f_cap = get_frac(simul_path, steps)*100 # %
        print(f'Percentage of atoms at the fiber: {f_cap:.2f} %')

        Nt = len(LoadTime(simul_path))
        n, rho_array, zeta_array = density(simul_path, rho_min=-1.5*RMOT/w0, rho_max=1.5*RMOT/w0, zeta_min=0, zeta_max=5, step=int(Nt/3))
        plot_density(simul_path, n, rho_array, zeta_array)
        _finalize_plot("density_contour")

        # plot_capfrac_vs_P(beam_name: str)
        # plot_capfrac_vs_P(beam=LGBeamL1())
        # plt.xlabel("Power (W)")
        # plt.ylabel("Final Captured Fraction (%)")
        # plt.title("Captured fraction vs Trapping PW")
        # plt.legend()
        # plt.grid()
        # plt.show()

        # plot_temperature(T, dMOT, chosen_beam)
        # plt.show()

        beam = Get_Beam(simul_path)
        CreateGif_desnity(simul_path, beam, beam_label=beam_name)
    except Exception as e:
        print(e)
