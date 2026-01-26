
### Using a 3D field grid as acceleration source

If you have a propagated field saved as `field_data.h5`, you can build a 3D
acceleration LUT and pass it to the Verlet solver:

```python
from FieldLUT import FieldLUT3D
from Verlet import verlet

field = FieldLUT3D.from_field_data_h5(
    "input/field_data.h5",
    acc_scale=1.0,
    max_points=200**3,
    dtype=np.float32,
    expected_axis_unit="r_F",
    expected_axis_scale=1.0,
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
want to integrate. The loader asserts that the requested domain is covered by
every z-slice and that the axis metadata matches expectations.
