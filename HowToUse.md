
### Using a 3D field grid as acceleration source

If you have a 3D Cartesian intensity grid saved as a `.npz` with arrays `x`, `y`, `z`
(1D axes) and `I` (shape `(Nx, Ny, Nz)`), you can build an acceleration LUT and
pass it to the Verlet solver:

```python
from FieldLUT import FieldLUT3D
from Verlet import verlet

field = FieldLUT3D.from_intensity_file(
    "path/to/field.npz",
    acc_scale=1.0,
    max_points=200**3,
    dtype=np.float32,
)

xs, vs, ts = verlet(
    x0=x0,
    v0=v0,
    a_func=field.acc,
    dt=dt,
    N_steps=N_steps,
    N_saves=N_saves,
    z_index=2,
)
```

`acc_scale` should convert the intensity gradient to the acceleration units you
want to integrate.
