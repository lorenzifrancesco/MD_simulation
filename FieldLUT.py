import numpy as np
from scipy.ndimage import map_coordinates


def _validate_uniform_axis(axis, name):
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


class FieldLUT3D:
    """
    3D lookup table for accelerations derived from an intensity grid.

    The LUT stores (ax, ay, az) on a uniform Cartesian grid. Interpolation uses
    linear sampling via scipy.ndimage.map_coordinates.
    """

    def __init__(self, x, y, z, ax, ay, az, acc_offset=None):
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
    ):
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

        return cls(x, y, z, ax, ay, az, acc_offset=acc_offset)

    def acc(self, pos):
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

        ix = (x_b - self.x0) / self.dx
        iy = (y_b - self.y0) / self.dy
        iz = (z_b - self.z0) / self.dz

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
