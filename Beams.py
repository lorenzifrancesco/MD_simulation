import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import sys
from Dynamics import *

class Beam:
    """
    Base class for optical beams.
    Defines shared initialization, scaling, and plotting utilities.
    """

    def __init__(self, name, P_b=1, lambda_b=1064e-9, w0_b=19e-6):
        self.name = name
        self.P_b = P_b
        self.lambda_b = lambda_b
        self.w0_b = w0_b

        # Cache-related attributes for analytic intensity(rho, zeta)
        self._use_intensity_cache = False
        self._rho_cache = None
        self._zeta_cache = None
        self._I_cache = None
        self._drho = None
        self._dzeta = None

        self.update_props()

    def update_props(self):
        """Update derived optical and scaling properties."""
        self.I0 = 4 * self.P_b / (np.pi * self.w0_b**2)
        self.zR = zR_func(self.lambda_b)
        self.alpha = alpha_func(self.lambda_b)
        self.phi = np.abs(self.alpha * self.I0 / (m_Rb * self.w0_b**2))
        self.tau = 1 / np.sqrt(self.phi)
        self.vs_rho = self.w0_b / self.tau
        self.vs_zeta = self.zR / self.tau
        self.as_rho = self.w0_b / self.tau**2
        self.as_zeta = self.zR / self.tau**2

    # --- setters ---
    def Set_Power(self, P_b: float):
        self.P_b = P_b
        self.update_props()

    def Set_Lambda(self, lambda_b: float):
        self.lambda_b = lambda_b
        self.update_props()

    def Set_w0(self, w0_b: float):
        self.w0_b = w0_b
        self.update_props()

    # --- generic beam geometry ---
    def w(self, z: float):
        """Beam radius as function of zeta (dimensionless)."""
        return np.sqrt(1 + z**2)

    def beta(self, zeta):
        return 1 / (1 + zeta**2)

    # --- intensity interface (to be overridden) ---
    def intensity_analytic(self, rho, zeta):
        """
        Analytic intensity profile I(rho, zeta) for the specific beam.
        Subclasses must override this.
        """
        raise NotImplementedError

    def intensity(self, rho, zeta):
        """
        Uses cached intensity when enabled, otherwise falls back to the
        analytic formula.
        """
        if self._use_intensity_cache:
            return self._intensity_from_cache(rho, zeta)
        return self.intensity_analytic(rho, zeta)

    # --- Intensity cache utilities -------------------------------------
    def enable_intensity_cache(
        self,
        rho_max=4.0,
        Nrho=256,
        zeta_min=-4.0,
        zeta_max=4.0,
        Nzeta=256,
    ):
        """
        Precompute a 2D cache for analytic intensity I(rho, zeta) on a
        regular grid and enable fast interpolation.

        Assumes cylindrical symmetry. TODO: implement also the asymmetric case.
        """
        rho = np.linspace(0.0, rho_max, Nrho)
        zeta = np.linspace(zeta_min, zeta_max, Nzeta)
        R, Z = np.meshgrid(rho, zeta, indexing="ij")
        I = self.intensity_analytic(R, Z)
        # print(sys.getsizeof(I)/(1e6)) # this is the size of the cache in MB
        
        self._rho_cache = rho
        self._zeta_cache = zeta
        self._I_cache = I
        self._drho = rho[1] - rho[0]
        self._dzeta = zeta[1] - zeta[0]
        self._use_intensity_cache = True

    def disable_intensity_cache(self):
        """Disable the analytic intensity cache and free the arrays."""
        self._use_intensity_cache = False
        self._rho_cache = None
        self._zeta_cache = None
        self._I_cache = None
        self._drho = None
        self._dzeta = None

    def _intensity_from_cache(self, rho, zeta):
        """
        Use scipy.ndimage.map_coordinates to interpolate from the cache.

        rho, zeta can be scalars or arrays; broadcasting is supported.
        Outside the cache domain we return 0 (via mode="constant").
        """
        if not self._use_intensity_cache or self._I_cache is None:
            raise RuntimeError("Intensity cache is not enabled")

        rho = np.asarray(rho)
        zeta = np.asarray(zeta)

        rho_b, zeta_b = np.broadcast_arrays(rho, zeta)
        shape = rho_b.shape

        rho_abs = np.abs(rho_b)
        rho0 = self._rho_cache[0]
        zeta0 = self._zeta_cache[0]

        # fractional indices in cache space
        ir = (rho_abs - rho0) / self._drho
        iz = (zeta_b - zeta0) / self._dzeta

        coords = np.vstack([ir.ravel(), iz.ravel()])
        I_flat = map_coordinates(
            self._I_cache,
            coords,
            order=1,          # linear interpolation
            mode="constant",  # outside grid -> cval
            cval=0.0,
            prefilter=False,  # faster for order=1
        )

        return I_flat.reshape(shape)

    # --- plotting utilities ---
    def plot_trans_intensity(self, x=np.linspace(-2, 2, 100), y=np.linspace(-2, 2, 100)):
        x, y = np.meshgrid(x, y)
        rho = np.sqrt(x**2 + y**2)
        cp = plt.contourf(x, y, self.intensity(rho, 0), levels=100, cmap="viridis")
        plt.colorbar(cp, label="Norm. Intensity")
        plt.xlabel(r"x ($w_0$)")
        plt.ylabel(r"y ($w_0$)")
        plt.title(f"Intensity contour plot {self.name} Beam")

    def plot_long_intensity(self, x=np.linspace(-2, 2, 100), z=np.linspace(0, 4, 100)):
        x, z = np.meshgrid(x, z)
        rho = np.abs(x)
        cp = plt.contourf(x, z, self.intensity(rho, z), levels=100, cmap="inferno", alpha=0.7)
        plt.colorbar(cp, label="Norm. Intensity")
        plt.xlabel(r"x ($w_0$)")
        plt.ylabel(r"z ($z_R$)")
        plt.title(f"Longitudinal Intensity contour plot {self.name} Beam")


# -------------------------------------------------------------------------
# Subclasses implementing specific beam profiles
# -------------------------------------------------------------------------

class GaussianBeam(Beam):
    """
    Gaussian optical beam (fundamental TEM00 mode).
    Provides dimensionless accelerations in (rho, zeta).
    """

    def __init__(self, P_b=1, lambda_b=1064e-9, w0_b=19e-6):
        super().__init__("Gauss", P_b, lambda_b, w0_b)
        # optional: precompute this constant once
        self._pref = self.lambda_b**2 / (np.pi**2 * self.w0_b**2)

    def intensity_analytic(self, rho, zeta):
        b = self.beta(zeta)
        return b * np.exp(-2 * b * rho**2)  # hot line

    # intensity() inherited from Beam:
    #   -> uses cache if enabled, otherwise intensity_analytic

    def acc(self, x):
        """
        Vectorized acceleration:
        x: shape (2, N) or (2,) with (rho, zeta).
        Returns array shape (2, N) with (a_rho, a_zeta).
        """
        rho, zeta = x

        # Compute beta and intensity ONCE
        b = self.beta(zeta)
        I = self.intensity(rho, zeta)   # single analytic or cache call

        # du/drho = 4 b rho I
        du_drho = 4.0 * b * rho * I

        # du/dzeta = pref * 2 b zeta (1 - 2 b rho^2) I
        # pref precomputed in __init__
        du_dzeta = self._pref * 2.0 * b * zeta * (1.0 - 2.0 * b * rho**2) * I

        acc_rho = -du_drho
        acc_zeta = -du_dzeta - g / self.as_zeta

        return np.vstack((acc_rho, acc_zeta))



class LGBeamL1(Beam):
    """
    Laguerre–Gaussian donut beam (p=0, ℓ=1).
    Provides dimensionless accelerations in (rho, zeta).
    """

    def __init__(self, P_b=1, lambda_b=532e-9, w0_b=19e-6):
        super().__init__("LG", P_b, lambda_b, w0_b)

    # intensity() inherited from Beam: cache if enabled, else analytic
    def intensity_analytic(self, rho, zeta):
        b = self.beta(zeta)
        s = 2 * b * rho**2
        return 2 * b * s * np.exp(-s)

    def du_drho(self, rho, zeta):
        rho = rho + 1e-5 * np.sign(rho)
        I = self.intensity(rho, zeta)
        b = self.beta(zeta)
        s = 2 * b * rho**2
        return I * (2 / rho) * (s - 1)

    def du_dzeta(self, rho, zeta):
        I = self.intensity(rho, zeta)
        b = self.beta(zeta)
        s = 2 * b * rho**2
        prefix = self.lambda_b**2 / (np.pi**2 * self.w0_b**2)
        return prefix * I * 2 * zeta * b * (2 - s)

    def acc(self, x):
        rho, zeta = x
        acc_rho = self.du_drho(rho, zeta)
        acc_zeta = self.du_dzeta(rho, zeta) - g / self.as_zeta
        return np.array([acc_rho, acc_zeta])


# -------------------------------------------------------------------------
# Beam dictionary for easy lookup
# -------------------------------------------------------------------------
beams = { # TODO it is probably a bit more robust to use an enum here
    "Gauss": GaussianBeam(),
    "LG": LGBeamL1()
}

# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: enable cache for Gaussian beam to speed intensity calls
    # -> with a very coarse grid to demonstrate functionality
    beams["Gauss"].enable_intensity_cache(
        rho_max=4.0,
        Nrho=5,
        zeta_min=0.0,
        zeta_max=4.0,
        Nzeta=5,
    )
    beams["LG"].enable_intensity_cache(
        rho_max=4.0,
        Nrho=5,
        zeta_min=0.0,
        zeta_max=4.0,
        Nzeta=5,
    )

    for name, beam in beams.items():
        beam.plot_trans_intensity()
        plt.show()
        beam.plot_long_intensity()
        plt.show()
        
    # ...and in full resolution
    
    beams["Gauss"].disable_intensity_cache()
    beams["LG"].disable_intensity_cache()

    for name, beam in beams.items():
        beam.plot_trans_intensity()
        plt.show()
        beam.plot_long_intensity()
        plt.show()
        
