# Bucket–Particle Force Calibration (Stage 1 Bug #2)

Standalone sweep that calibrated `soil.bucket_push_force` to a physically reasonable value, after the original Stage 1 code path was discovered to use `contact_damping = 1000 N` directly as the bucket push force — a unit-confusion bug (the field's documented unit is N·s/m, not N) that ejected particles at ~3 km/s.

## Method

A 200-particle heap is created in the standard Warp particle system, settled for 100 steps under gravity, then a virtual bucket sweeps horizontally through it for 20 steps at +0.4 m/s with `bucket_normal = [+1, 0, 0]`. The displacement of every particle is recorded for a sweep of `push_strength` ∈ {0.01, 0.1, 1.0, 10.0, 1000.0} N. Friction coefficient is held at 0.05 to isolate the push-force effect.

The bucket geometry, particle radius (1 cm), particle mass (≈ 6.3 mg), and substeps match the production env config.

## Results

| `push_strength` (N) | Particles moved > 1 cm | Mean Δx | Max displacement | Regime |
|---|---|---|---|---|
| 0.01 | 200 / 200 | +39 mm | 46 mm | bucket barely registers; particles slide back after pass |
| 0.10 | 200 / 200 | +167 mm | 203 mm | partial drag; particles trail behind bucket |
| **1.00** | **200 / 200** | **+366 mm** | **463 mm** | **carry regime — particles travel with bucket (~ 0.4 m sweep, ~ 0.4 m mean Δx)** |
| 10.00 | 200 / 200 | +910 mm | 910 mm | severe overshoot; particles outrun bucket |
| 1000.00 | 200 / 200 | **+90 863 mm** | **90 863 mm** | **ballistic ejection — original buggy default** |

## Conclusion

`bucket_push_force = 1.0 N` was selected as the production default. It is the only setting where the per-particle displacement matches the bucket's own travel distance, i.e. the bucket *carries* particles instead of either *missing* them (low strength) or *launching* them off the map (high strength). The chosen value is also physically sensible relative to gravity: a 6.3 mg particle experiences ~ 0.06 N from gravity, so 1 N produces accelerations on the order of 16 g — strong enough to overcome rest friction without ballistic behaviour.

The fix in code:

- `soil/soil_properties.py:54` — added explicit `bucket_push_force: float = 1.0` field
- `soil/particle_system.py:248` — kernel now reads `self.soil.bucket_push_force` (was `self.soil.contact_damping`)
