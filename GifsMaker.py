import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from Beams import *

def MakeGif_density(
    pos: np.ndarray,
    density: np.ndarray,
    beam=GaussianBeam(),
    file_name='density_vs_time',
    intensity_grid=None,
    beam_label=None,
):
    """
    Create a GIF animation of atomic density over time.

    Parameters:
    ----------
    file_name : str, optional
        Name of the output GIF file.
    """

    rho_array, zeta_array = pos

    R, Z = np.meshgrid(rho_array * w0 * 1e3, zeta_array * beam.zR * 1e3)

    fig, ax = plt.subplots(figsize=(7, 10))

    density_max = float(np.max(density))
    if density_max <= 0.0:
        density_max = 1.0

    # --- Density overlay with transparency ---
    img_density = ax.imshow(
        density[0],
        extent=[R.min(), R.max(), Z.min(), Z.max()],
        origin='lower',
        cmap='viridis',
        aspect='auto',
        vmin=0.0,
        vmax=density_max,
    )
    
    # --- Beam intensity as background ---
    if intensity_grid is None:
        rho_dim = R / (w0 * 1e3)
        zeta_dim = Z / (beam.zR * 1e3)
        I = beam.intensity(rho_dim, zeta_dim)
    else:
        I = np.asarray(intensity_grid, dtype=float)
    I_max = float(np.max(I))
    if I_max <= 0.0:
        I_max = 1.0
    I = I / I_max
    
    # img_intensity = ax.imshow(I, extent=[R.min(), R.max(), Z.min(), Z.max()],
    #                           origin='lower', cmap='inferno', aspect='auto', alpha=0.1)
    
    # overlay with alpha
    cmap = plt.cm.inferno
    levels = np.linspace(0.0, 1.0, 50)
    cf = ax.contourf(R, Z, I, levels=levels, cmap=cmap, alpha=0.1)

    # make a mappable for the colorbar with opaque colors
    sm = mpl.cm.ScalarMappable(norm=cf.norm, cmap=cmap)
    sm.set_array([])  
    fig.colorbar(sm, ax=ax, label="Beam intensity")

    label = beam_label if beam_label is not None else beam.name
    ax.set_title(f'Density and Intensity distribution ({label})')
    ax.set_xlabel(r'$\rho$ (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_ylim(0, Z.max())

    # Colorbars
    fig.colorbar(img_density, ax=ax, label="Atomic Density")
    # fig.colorbar(img_intensity, ax=ax, label="Beam Intensity")

    # --- Animation update ---
    def update(i):
        img_density.set_data(density[i])
        return [img_density]

    ani = FuncAnimation(fig, update, frames=int(0.7*len(density)), interval=100)
    ani.save('./gifs/' + file_name + '.gif', writer='pillow', fps=10)
    plt.close(fig)
