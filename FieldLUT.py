import numpy as np
from scipy.ndimage import map_coordinates
from scipy.constants import g
import h5py

# =================
# Utils
# =================
def _validate_uniform_axis(axis, name):
    """Validate that an axis is 1D, uniformly spaced, and strictly increasing."""
    axis = np.asarray(axis, dtype=float)
    if axis.ndim != 1:
        raise ValueError(f"{name} axis must be 1D")
    if axis.size < 2:
        raise ValueError(f"{name} axis must have at least 2 points")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"{name} axis must be strictly increasing")
    step = axis[1] - axis[0]
    if not np.allclose(np.diff(axis), step, rtol=1e-6, atol=1e-12):
        raise ValueError(f"{name} axis must be uniformly spaced")
    return axis, step


def _normalize_stride(stride):
    if stride is None:
        return 1, 1, 1
    if np.isscalar(stride):
        s = int(stride)
        return s, s, s
    if len(stride) != 3:
        raise ValueError("stride must be an int or a 3-tuple")
    return tuple(int(s) for s in stride)


def _stride_for_max_points(nx, ny, nz, max_points):
    if max_points is None:
        return 1
    total = nx * ny * nz
    if total <= max_points:
        return 1
    factor = int(np.ceil((total / max_points) ** (1.0 / 3.0)))
    return max(1, factor)


def _axis_tolerance(axis):
    """Match the tolerance to the axis max"""
    scale = float(np.max(np.abs(axis))) if np.size(axis) else 1.0
    return max(1e-9, 1e-8 * scale) # this hardcodes the tolerance!


def _grid_indices(values, origin, step, size):
    idx = (values - origin) / step
    tol = 1e-9 * max(1.0, float(size))
    on_grid = (idx >= -tol) & (idx <= (size - 1) + tol)
    return np.where(on_grid, np.clip(idx, 0.0, size - 1), idx)


def _aligned_axis(start, stop, step, name):
    if step <= 0.0:
        raise ValueError(f"{name} step must be positive")
    span = float(stop - start)
    n_float = span / step
    n_round = int(round(n_float))
    if not np.isclose(n_float, n_round, rtol=1e-6, atol=1e-9):
        raise ValueError(
            f"{name} domain [{start}, {stop}] is not aligned with step {step}"
        )
    axis = start + step * np.arange(n_round + 1)
    return axis




class FieldLUT3D:
    """
    3D lookup table for accelerations derived from an H5 intensity grid.

    The LUT stores vector acceleration (ax, ay, az) for Verlet MD on a uniform Cartesian grid. 
    Interpolation uses linear sampling via scipy.ndimage.map_coordinates.
    """

    def __init__(self, x, y, z, ax, ay, az, acc_offset=None, intensity=None):
        self.x, self.dx = _validate_uniform_axis(x, "x")
        self.y, self.dy = _validate_uniform_axis(y, "y")
        self.z, self.dz = _validate_uniform_axis(z, "z")

        self.ax = np.asarray(ax)
        self.ay = np.asarray(ay)
        self.az = np.asarray(az)

        expected_shape = (self.x.size, self.y.size, self.z.size)
        if self.ax.shape != expected_shape:
            raise ValueError("ax shape does not match grid axes")
        if self.ay.shape != expected_shape:
            raise ValueError("ay shape does not match grid axes")
        if self.az.shape != expected_shape:
            raise ValueError("az shape does not match grid axes")

        if intensity is None:
            self.intensity = None
        else:
            intensity = np.asarray(intensity)
            if intensity.shape != expected_shape:
                raise ValueError("intensity shape does not match grid axes")
            self.intensity = intensity

        if acc_offset is None:
            self.acc_offset = None
        else:
            offset = np.asarray(acc_offset, dtype=float)
            if offset.shape != (3,):
                raise ValueError("acc_offset must be a length-3 vector")
            self.acc_offset = offset

        self.x0 = self.x[0]
        self.y0 = self.y[0]
        self.z0 = self.z[0]

    def _sample_intensity(self, x, y, z):
        if self.intensity is None:
            raise ValueError(
                "FieldLUT3D does not store intensity; load it with keep_intensity=True"
            )

        x_b, y_b, z_b = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )
        shape = x_b.shape

        ix = _grid_indices(x_b, self.x0, self.dx, self.x.size)
        iy = _grid_indices(y_b, self.y0, self.dy, self.y.size)
        iz = _grid_indices(z_b, self.z0, self.dz, self.z.size)
        coords = np.vstack([ix.ravel(), iy.ravel(), iz.ravel()])

        return map_coordinates(
            self.intensity,
            coords,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).reshape(shape)

    def intensity_grid(self, rho_array, zeta_array, beam, r_ref):
        s_r = beam.w0_b / r_ref
        s_z = beam.zR / r_ref

        rho_array = np.asarray(rho_array, dtype=float)
        zeta_array = np.asarray(zeta_array, dtype=float)
        x_field = np.abs(rho_array) * s_r
        z_field = zeta_array * s_z

        x_grid, z_grid = np.meshgrid(x_field, z_field, indexing="xy")
        return self._sample_intensity(x_grid, 0.0, z_grid)

    def intensity_planes(self, x_axis, y_axis, z_axis):
        x_axis = np.asarray(x_axis, dtype=float)
        y_axis = np.asarray(y_axis, dtype=float)
        z_axis = np.asarray(z_axis, dtype=float)

        x_grid, z_grid = np.meshgrid(x_axis, z_axis, indexing="ij")
        y_grid, z_grid_y = np.meshgrid(y_axis, z_axis, indexing="ij")
        intensity_xz = self._sample_intensity(x_grid, 0.0, z_grid)
        intensity_yz = self._sample_intensity(0.0, y_grid, z_grid_y)
        return intensity_xz, intensity_yz

    @classmethod
    def acceleration_from_field_data_h5(
        cls,
        path,
        beam,
        r_ref,
        expected_axis_unit="r_F",
        expected_axis_scale=1.0,
        acc_scale=1.0,
        required_rho=None,
        required_zeta=None,
        auto_scale=False,
        grid_pad=0.02,
        outside_mode="gravity",
        warn_outside=True,
        keep_intensity=True,
    ):
        field = cls.from_field_data_h5(
            path=path,
            expected_axis_unit=expected_axis_unit,
            expected_axis_scale=expected_axis_scale,
            acc_scale=acc_scale,
            keep_intensity=keep_intensity,
        )
        field_unit = expected_axis_unit

        r_ref = float(r_ref)
        if not np.isfinite(r_ref) or r_ref <= 0.0:
            raise ValueError("r_ref must be a positive finite number")

        if field.y[0] > 0.0 or field.y[-1] < 0.0:
            raise ValueError("LUT y-axis does not include 0; cannot slice at y=0")

        grid_pad = float(grid_pad)
        if grid_pad < 0.0 or grid_pad >= 1.0:
            raise ValueError("grid_pad must be in [0, 1)")

        r_ref_min = r_ref
        x_lim = None
        if field.x[0] < 0.0 < field.x[-1]:
            x_lim = min(-field.x[0], field.x[-1]) * (1.0 - grid_pad)

        if required_rho is not None:
            rho_min, rho_max = (float(required_rho[0]), float(required_rho[1]))
            max_abs_rho = max(abs(rho_min), abs(rho_max))
            if x_lim is None:
                raise ValueError("LUT x-axis does not straddle 0; cannot map rho")
            if max_abs_rho > 0.0:
                r_ref_min = max(r_ref_min, beam.w0_b * max_abs_rho / x_lim)

        if required_zeta is not None:
            zeta_min, zeta_max = (float(required_zeta[0]), float(required_zeta[1]))
            if zeta_min < 0.0 and field.z[0] >= 0.0:
                raise ValueError("LUT z-axis does not include negative zeta values")
            max_abs_zeta = max(abs(zeta_min), abs(zeta_max))
            if max_abs_zeta > 0.0:
                z_lim = max(abs(field.z[0]), abs(field.z[-1])) * (1.0 - grid_pad)
                r_ref_min = max(r_ref_min, beam.zR * max_abs_zeta / z_lim)

        if r_ref_min > r_ref * (1.0 + 1e-6):
            if auto_scale:
                r_ref_min *= 1.0 + grid_pad
                print(
                    "WARNING: LUT coverage too small; increasing r_ref from "
                    f"{r_ref:.3e} to {r_ref_min:.3e} to fit domain."
                )
                r_ref = r_ref_min
            else:
                print(
                    "WARNING: LUT coverage too small for requested domain. "
                    "Proceeding without rescaling; outside mode will apply."
                )

        s_r = beam.w0_b / r_ref
        s_z = beam.zR / r_ref
        if s_r <= 0.0 or s_z <= 0.0:
            raise ValueError("Invalid scale factors for LUT mapping")

        tol_x = 1e-6 * max(1.0, float(np.max(np.abs(field.x))))
        tol_z = 1e-6 * max(1.0, float(np.max(np.abs(field.z))))

        outside_mode = str(outside_mode).lower()
        if outside_mode not in ("zero", "gravity", "error"):
            raise ValueError("outside_mode must be 'zero', 'gravity', or 'error'")
        warned = False

        def acc(x):
            nonlocal warned
            rho, zeta = x
            x_field = rho * s_r
            z_field = zeta * s_z

            inside = (
                (x_field >= field.x[0] - tol_x)
                & (x_field <= field.x[-1] + tol_x)
                & (z_field >= field.z[0] - tol_z)
                & (z_field <= field.z[-1] + tol_z)
            )

            if outside_mode == "error" and np.any(~inside):
                x_min = float(np.min(x_field))
                x_max = float(np.max(x_field))
                z_min = float(np.min(z_field))
                z_max = float(np.max(z_field))
                raise ValueError(
                    "LUT coverage too small for requested domain "
                    f"(x_field {x_min:.3g}..{x_max:.3g} vs "
                    f"{field.x[0]:.3g}..{field.x[-1]:.3g}, "
                    f"z_field {z_min:.3g}..{z_max:.3g} vs "
                    f"{field.z[0]:.3g}..{field.z[-1]:.3g})"
                )
            if warn_outside and not warned and np.any(~inside):
                warned = True
                print("WARNING: particles outside LUT domain, using fallback mode.")

            pos = np.vstack([x_field, np.zeros_like(x_field), z_field])
            acc_cart = field.acc(pos)
            acc_rho = acc_cart[0] / s_r
            acc_zeta = acc_cart[2] / s_z
            if outside_mode == "zero":
                acc_rho = np.where(inside, acc_rho, 0.0)
                acc_zeta = np.where(inside, acc_zeta, 0.0)
                acc_zeta = acc_zeta - g / beam.as_zeta * inside
            elif outside_mode == "gravity":
                acc_zeta = acc_zeta - g / beam.as_zeta
                acc_zeta = np.where(inside, acc_zeta, -g / beam.as_zeta)
            else:
                acc_zeta = acc_zeta - g / beam.as_zeta
            return np.vstack([acc_rho, acc_zeta])

        rho_min, rho_max = required_rho if required_rho is not None else (np.nan, np.nan)
        zeta_min, zeta_max = required_zeta if required_zeta is not None else (np.nan, np.nan)
        lut_info = {
            "grid": (field.x.size, field.y.size, field.z.size),
            "field": field,
            "axis_unit": field_unit,
            "x_range": (float(field.x[0]), float(field.x[-1])),
            "y_range": (float(field.y[0]), float(field.y[-1])),
            "z_range": (float(field.z[0]), float(field.z[-1])),
            "rho_range": (float(rho_min), float(rho_max)),
            "zeta_range": (float(zeta_min), float(zeta_max)),
            "x_req_range": (float(rho_min * s_r), float(rho_max * s_r)),
            "z_req_range": (float(zeta_min * s_z), float(zeta_max * s_z)),
            "r_ref": float(r_ref),
            "s_r": float(s_r),
            "s_z": float(s_z),
            "outside_mode": outside_mode,
        }

        return acc, lut_info

    @classmethod
    def from_intensity_file(
        cls,
        path,
        acc_scale=1.0,
        max_points=None,
        stride=None,
        dtype=np.float32,
        x_key="x",
        y_key="y",
        z_key="z",
        I_key="I",
        acc_offset=None,
        keep_intensity=False,
    ):
        """Load the intensity file"""
        path_str = str(path)
        if path_str.lower().endswith((".h5", ".hdf5")):
            return cls.from_field_data_h5(
                path=path,
                acc_scale=acc_scale,
                max_points=max_points,
                stride=stride,
                dtype=dtype,
                acc_offset=acc_offset,
                keep_intensity=keep_intensity,
            )
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            try:
                x = data[x_key]
                y = data[y_key]
                z = data[z_key]
                intensity = data[I_key]
            finally:
                data.close()
        else:
            raise ValueError("Expected a .npz file with x, y, z, I arrays")
        return cls.from_intensity_grid(
            x=x,
            y=y,
            z=z,
            intensity=intensity,
            acc_scale=acc_scale,
            max_points=max_points,
            stride=stride,
            dtype=dtype,
            acc_offset=acc_offset,
            keep_intensity=keep_intensity,
        )

    @classmethod
    def from_field_data_h5(
        cls,
        path,
        acc_scale=1.0,
        max_points=None,
        stride=None,
        dtype=np.float32,
        acc_offset=None,
        x_domain=None,
        y_domain=None,
        z_domain=None,
        expected_axis_unit=None,
        expected_axis_scale=None,
        axis_scale=None,
        apply_axis_scale=True,
        intensity_key="intensity",
        keep_intensity=False,
    ):
        """Load the LUT from a field_data.h5 file with the expected structure and metadata.
        Apply consistency checks"""
        if h5py is None:
            raise ImportError("h5py is required to load .h5 field data")

        with h5py.File(path, "r") as f:
            if "domain/z" not in f:
                raise ValueError("Missing domain/z dataset in HDF5 file")

            axis_unit = f.attrs.get("axis_unit", None)
            axis_scale_attr = f.attrs.get("axis_scale", 1.0)
            if expected_axis_unit is not None and axis_unit != expected_axis_unit:
                raise ValueError(
                    f"axis_unit mismatch: expected {expected_axis_unit}, got {axis_unit}"
                )
            if expected_axis_scale is not None and not np.isclose(
                axis_scale_attr, expected_axis_scale, rtol=1e-6, atol=1e-12
            ):
                raise ValueError(
                    f"axis_scale mismatch: expected {expected_axis_scale}, got {axis_scale_attr}"
                )

            if axis_scale is None:
                axis_scale = axis_scale_attr
            axis_scale = float(axis_scale)
            if not np.isfinite(axis_scale) or axis_scale <= 0.0:
                raise ValueError("axis_scale must be a positive finite number")

            scale = axis_scale if apply_axis_scale else 1.0

            z_axis = np.asarray(f["domain/z"][:], dtype=float) * scale
            z_axis, _ = _validate_uniform_axis(z_axis, "z")

            fields_group = f.get("fields")
            if fields_group is None:
                raise ValueError("Missing fields group in HDF5 file")

            slice_names = sorted(fields_group.keys())
            if not slice_names:
                raise ValueError("No z-slices found in fields group")

            indices = []
            for name in slice_names:
                if not name.startswith("z_"):
                    continue
                try:
                    indices.append(int(name.split("_", 1)[1]))
                except ValueError as exc:
                    raise ValueError(f"Unexpected slice name: {name}") from exc
            if len(indices) != len(z_axis):
                raise ValueError(
                    "Number of z-slices does not match domain/z axis length"
                )
            indices = sorted(indices)
            if indices != list(range(len(z_axis))):
                raise ValueError("z-slices are not contiguous from 0 to Nz-1")

            dx_attr = f.attrs.get("dx", None)
            x_min = -np.inf
            x_max = np.inf
            y_min = -np.inf
            y_max = np.inf
            x_mins = []
            x_maxs = []
            y_mins = []
            y_maxs = []
            dxs = []
            dys = []
            slice_meta = []

            for idx in indices:
                g = fields_group[f"z_{idx:05d}"]
                if intensity_key not in g:
                    raise ValueError(
                        f"Missing {intensity_key} dataset in slice z_{idx:05d}"
                    )
                x = np.asarray(g["x"][:], dtype=float) * scale
                y = np.asarray(g["y"][:], dtype=float) * scale
                intensity = np.asarray(g[intensity_key][:], dtype=float)

                x, dx = _validate_uniform_axis(x, "x")
                y, dy = _validate_uniform_axis(y, "y")
                if intensity.shape != (x.size, y.size):
                    raise ValueError(
                        f"intensity shape mismatch in slice z_{idx:05d}"
                    )

                x_min = max(x_min, x[0])
                x_max = min(x_max, x[-1])
                y_min = max(y_min, y[0])
                y_max = min(y_max, y[-1])
                x_mins.append(x[0])
                x_maxs.append(x[-1])
                y_mins.append(y[0])
                y_maxs.append(y[-1])
                dxs.append(dx)
                dys.append(dy)
                slice_meta.append((x, y, intensity))

            if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min >= x_max:
                raise ValueError("No common x-domain across all z-slices")
            if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min >= y_max:
                raise ValueError("No common y-domain across all z-slices")

            if x_domain is None:
                x_domain = (x_min, x_max)
            if y_domain is None:
                y_domain = (y_min, y_max)

            x_domain = tuple(float(v) for v in x_domain)
            y_domain = tuple(float(v) for v in y_domain)
            if x_domain[0] >= x_domain[1]:
                raise ValueError("x_domain must be (min, max)")
            if y_domain[0] >= y_domain[1]:
                raise ValueError("y_domain must be (min, max)")
            if x_domain[0] < x_min - _axis_tolerance([x_min, x_max]) or x_domain[
                1
            ] > x_max + _axis_tolerance([x_min, x_max]):
                raise ValueError("Requested x_domain exceeds HDF5 coverage")
            if y_domain[0] < y_min - _axis_tolerance([y_min, y_max]) or y_domain[
                1
            ] > y_max + _axis_tolerance([y_min, y_max]):
                raise ValueError("Requested y_domain exceeds HDF5 coverage")

            if z_domain is not None:
                z_domain = tuple(float(v) for v in z_domain)
                if z_domain[0] >= z_domain[1]:
                    raise ValueError("z_domain must be (min, max)")
                if z_domain[0] < z_axis[0] - _axis_tolerance(z_axis) or z_domain[
                    1
                ] > z_axis[-1] + _axis_tolerance(z_axis):
                    raise ValueError("Requested z_domain exceeds HDF5 coverage")
                z_start_matches = np.isclose(
                    z_axis, z_domain[0], rtol=1e-6, atol=_axis_tolerance(z_axis)
                )
                z_stop_matches = np.isclose(
                    z_axis, z_domain[1], rtol=1e-6, atol=_axis_tolerance(z_axis)
                )
                if not z_start_matches.any() or not z_stop_matches.any():
                    raise ValueError("z_domain boundaries must align to z grid")
                z_start = int(np.where(z_start_matches)[0][0])
                z_stop = int(np.where(z_stop_matches)[0][0])
                if z_stop <= z_start:
                    raise ValueError("z_domain must span at least 2 z points")
                z_indices = list(range(z_start, z_stop + 1))
                z_axis = z_axis[z_start : z_stop + 1]
            else:
                z_indices = list(range(len(z_axis)))

            if len(z_axis) < 2:
                raise ValueError("z axis must contain at least 2 points")

            dxs = np.asarray(dxs, dtype=float)
            dys = np.asarray(dys, dtype=float)
            x_mins = np.asarray(x_mins, dtype=float)
            x_maxs = np.asarray(x_maxs, dtype=float)
            y_mins = np.asarray(y_mins, dtype=float)
            y_maxs = np.asarray(y_maxs, dtype=float)

            x_span = x_maxs - x_mins
            y_span = y_maxs - y_mins
            x_tight_idx = int(np.argmin(x_span))
            y_tight_idx = int(np.argmin(y_span))
            dx_ref = float(dxs[x_tight_idx])
            dy_ref = float(dys[y_tight_idx])
            if dx_ref <= 0.0 or dy_ref <= 0.0:
                raise ValueError("Invalid dx in HDF5 metadata")
            dx_ref *= scale
            dy_ref *= scale

            if dx_attr is not None:
                dx_attr_scaled = float(dx_attr) * scale
                if not np.allclose(dxs, dx_attr_scaled, rtol=2e-2, atol=1e-6):
                    raise ValueError("Per-slice dx does not match HDF5 dx")
                if not np.allclose(dys, dx_attr_scaled, rtol=2e-2, atol=1e-6):
                    raise ValueError("Per-slice dy does not match HDF5 dx")
                if not np.isclose(dx_ref, dx_attr_scaled, rtol=2e-2, atol=1e-6):
                    raise ValueError("x-axis spacing is inconsistent with HDF5 dx")
                if not np.isclose(dy_ref, dx_attr_scaled, rtol=2e-2, atol=1e-6):
                    raise ValueError("y-axis spacing is inconsistent with HDF5 dx")
            else:
                if (np.max(dxs) - np.min(dxs)) / np.median(dxs) > 0.02:
                    raise ValueError("Per-slice dx varies too much to merge safely")
                if (np.max(dys) - np.min(dys)) / np.median(dys) > 0.02:
                    raise ValueError("Per-slice dy varies too much to merge safely")

            x_axis = _aligned_axis(x_domain[0], x_domain[1], dx_ref, "x")
            y_axis = _aligned_axis(y_domain[0], y_domain[1], dy_ref, "y")
            if x_axis.size < 2 or y_axis.size < 2:
                raise ValueError("x/y axis must have at least 2 points")

            intensity_grid = np.empty(
                (x_axis.size, y_axis.size, len(z_axis)), dtype=float
            )

            for out_k, idx in enumerate(z_indices):
                x, y, intensity = slice_meta[idx]
                x_tol = _axis_tolerance(x)
                y_tol = _axis_tolerance(y)
                if x_axis[0] < x[0] - x_tol or x_axis[-1] > x[-1] + x_tol:
                    raise ValueError("x_axis exceeds coverage in one or more slices")
                if y_axis[0] < y[0] - y_tol or y_axis[-1] > y[-1] + y_tol:
                    raise ValueError("y_axis exceeds coverage in one or more slices")

                ix = (x_axis - x[0]) / (x[1] - x[0])
                iy = (y_axis - y[0]) / (y[1] - y[0])
                grid_x, grid_y = np.meshgrid(ix, iy, indexing="ij")
                coords = np.vstack([grid_x.ravel(), grid_y.ravel()])
                resampled = map_coordinates(
                    intensity,
                    coords,
                    order=1,
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                ).reshape(x_axis.size, y_axis.size)
                intensity_grid[:, :, out_k] = resampled

        return cls.from_intensity_grid(
            x=x_axis,
            y=y_axis,
            z=z_axis,
            intensity=intensity_grid,
            acc_scale=acc_scale,
            max_points=max_points,
            stride=stride,
            dtype=dtype,
            acc_offset=acc_offset,
            keep_intensity=keep_intensity,
        )

    @classmethod
    def from_intensity_grid(
        cls,
        x,
        y,
        z,
        intensity,
        acc_scale=1.0,
        max_points=None,
        stride=None,
        dtype=np.float32,
        acc_offset=None,
        keep_intensity=False,
    ):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        intensity = np.asarray(intensity, dtype=float)

        if intensity.shape != (x.size, y.size, z.size):
            raise ValueError("intensity shape must be (Nx, Ny, Nz)")

        auto_stride = _stride_for_max_points(x.size, y.size, z.size, max_points)
        sx, sy, sz = _normalize_stride(stride)
        sx = max(sx, auto_stride)
        sy = max(sy, auto_stride)
        sz = max(sz, auto_stride)

        if sx > 1 or sy > 1 or sz > 1:
            x = x[::sx]
            y = y[::sy]
            z = z[::sz]
            intensity = intensity[::sx, ::sy, ::sz]

        if x.size < 2 or y.size < 2 or z.size < 2:
            raise ValueError("Downsampling left an axis with < 2 points")

        grad = np.gradient(intensity, x, y, z, edge_order=1)
        acc_scale = np.asarray(acc_scale, dtype=float)
        if acc_scale.shape == ():
            scale = np.array([acc_scale, acc_scale, acc_scale], dtype=float)
        elif acc_scale.shape == (3,):
            scale = acc_scale
        else:
            raise ValueError("acc_scale must be a scalar or length-3 vector")

        ax = (grad[0] * scale[0]).astype(dtype, copy=False)
        ay = (grad[1] * scale[1]).astype(dtype, copy=False)
        az = (grad[2] * scale[2]).astype(dtype, copy=False)

        stored_intensity = None
        if keep_intensity:
            stored_intensity = intensity.astype(dtype, copy=False)

        return cls(
            x,
            y,
            z,
            ax,
            ay,
            az,
            acc_offset=acc_offset,
            intensity=stored_intensity,
        )

    def acc(self, pos):
        """ Compute the interpolation for the acceleration. May run hot"""
        pos = np.asarray(pos, dtype=float)
        transpose = False
        if pos.ndim == 1:
            if pos.shape[0] != 3:
                raise ValueError("pos must be length-3 for a single point")
            pos = pos[:, None]
        elif pos.ndim == 2 and pos.shape[0] != 3 and pos.shape[1] == 3:
            pos = pos.T
            transpose = True
        elif pos.ndim != 2 or pos.shape[0] != 3:
            raise ValueError("pos must have shape (3, N) or (N, 3)")

        x, y, z = pos
        x_b, y_b, z_b = np.broadcast_arrays(x, y, z)
        shape = x_b.shape

        ix = _grid_indices(x_b, self.x0, self.dx, self.x.size)
        iy = _grid_indices(y_b, self.y0, self.dy, self.y.size)
        iz = _grid_indices(z_b, self.z0, self.dz, self.z.size)

        coords = np.vstack([ix.ravel(), iy.ravel(), iz.ravel()])

        ax = map_coordinates(
            self.ax,
            coords,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        ay = map_coordinates(
            self.ay,
            coords,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        az = map_coordinates(
            self.az,
            coords,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )

        acc = np.vstack([ax, ay, az]).reshape(3, *shape)
        if self.acc_offset is not None:
            offset = self.acc_offset.reshape(3, *([1] * len(shape)))
            acc = acc + offset

        if transpose:
            return acc.T
        return acc

    def plot_acc_slices(
        self,
        x0=0.0,
        y0=0.0,
        planes=("xz", "yz"),
        cmap="coolwarm",
        show=True,
        use_tex=False,
    ):
        """
        Plot acceleration component slices on two planes through the central axis.

        By default this renders xz (at y=y0) and yz (at x=x0) slices for ax, ay, az.
        """
        import matplotlib.pyplot as plt

        if use_tex is not None:
            plt.rcParams["text.usetex"] = bool(use_tex)

        if len(planes) != 2:
            raise ValueError("planes must be a 2-tuple like ('xz', 'yz')")
        planes = tuple(planes)
        for plane in planes:
            if plane not in ("xz", "yz", "xy"):
                raise ValueError("planes entries must be 'xz', 'yz', or 'xy'")

        ix0 = int(np.argmin(np.abs(self.x - x0)))
        iy0 = int(np.argmin(np.abs(self.y - y0)))

        components = [("ax", self.ax), ("ay", self.ay), ("az", self.az)]
        fig, axes = plt.subplots(
            nrows=len(components),
            ncols=len(planes),
            figsize=(5.0 * len(planes), 3.5 * len(components)),
            constrained_layout=True,
        )

        if axes.ndim == 1:
            axes = axes.reshape(len(components), len(planes))

        for row, (label, comp) in enumerate(components):
            for col, plane in enumerate(planes):
                ax = axes[row, col]
                if plane == "xz":
                    X, Z = np.meshgrid(self.x, self.z, indexing="ij")
                    data = comp[:, iy0, :]
                    mesh = ax.pcolormesh(X, Z, data, shading="auto", cmap=cmap)
                    ax.set_xlabel("x")
                    ax.set_ylabel("z")
                    ax.set_title(f"{label} on xz @ y={self.y[iy0]:.3g}")
                elif plane == "yz":
                    Y, Z = np.meshgrid(self.y, self.z, indexing="ij")
                    data = comp[ix0, :, :]
                    mesh = ax.pcolormesh(Y, Z, data, shading="auto", cmap=cmap)
                    ax.set_xlabel("y")
                    ax.set_ylabel("z")
                    ax.set_title(f"{label} on yz @ x={self.x[ix0]:.3g}")
                else:
                    X, Y = np.meshgrid(self.x, self.y, indexing="ij")
                    data = comp[:, :, int(self.z.size // 2)]
                    mesh = ax.pcolormesh(X, Y, data, shading="auto", cmap=cmap)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_title(
                        f"{label} on xy @ z={self.z[int(self.z.size // 2)]:.3g}"
                    )

                fig.colorbar(mesh, ax=ax, label=label)

        if show:
            plt.show()
        return fig, axes


if __name__ == "__main__":
    # Loading example / check that the data are correct, using actual field:
    import argparse
    import os
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Render LUT acceleration slices from a field_data.h5 file."
    )
    parser.add_argument(
        "--path",
        default="input/field_data.h5",
        help="Path to field_data.h5 (default: input/field_data.h5)",
    )
    parser.add_argument("--x0", type=float, default=0.0, help="Slice x0 for yz plane.")
    parser.add_argument("--y0", type=float, default=0.0, help="Slice y0 for xz plane.")
    parser.add_argument(
        "--outdir",
        default="media/lut",
        help="Output directory for plots (default: media/lut)",
    )
    parser.add_argument(
        "--planes",
        nargs=2,
        default=("xz", "yz"),
        help="Two planes to plot, e.g. xz yz",
    )
    args = parser.parse_args()

    field = FieldLUT3D.from_intensity_file(args.path)
    fig, _ = field.plot_acc_slices(
        x0=args.x0,
        y0=args.y0,
        planes=tuple(args.planes),
        show=False,
    )

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "lut_acc_slices.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved LUT slices to {out_path}")
