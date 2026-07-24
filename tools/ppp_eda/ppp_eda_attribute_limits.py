"""Shared fixed display limits for physical PPP radar attributes.

These limits affect visualization only. All finite observations remain in the
summary statistics. Established target attributes use rounded empirical tail
limits from the thesis. Newly exported intermediate velocity stages use their
empirical 1% and 99% quantiles at plotting time until stable rounded limits are
chosen from a full scan.
"""

from __future__ import annotations

from typing import Dict, Tuple


DEPTH_DISPLAY_RANGE: Tuple[float, float] = (0.0, 170.0)
VELOCITY_DISPLAY_RANGE: Tuple[float, float] = (-7.0, 7.0)
RCS_DISPLAY_RANGE: Tuple[float, float] = (-50.0, 20.0)

RANGE_GATE_LENGTH_DISPLAY_RANGE: Tuple[float | None, float | None] = (None, None)
DOPPLER_RANGE_DISPLAY_RANGE: Tuple[float | None, float | None] = (None, None)

# None means: resolve one fixed range from the complete scan using q01/q99.
VELOCITY_MEASURED_RELATIVE_DISPLAY_RANGE: Tuple[float | None, float | None] = (
    None,
    None,
)
VELOCITY_EXPECTED_STATIC_DISPLAY_RANGE: Tuple[float | None, float | None] = (
    None,
    None,
)
VELOCITY_COMPENSATED_PRE_WRAP_DISPLAY_RANGE: Tuple[float | None, float | None] = (
    None,
    None,
)
EGO_LONGITUDINAL_VELOCITY_DISPLAY_RANGE: Tuple[float | None, float | None] = (
    None,
    None,
)
EGO_YAW_RATE_DISPLAY_RANGE: Tuple[float | None, float | None] = (None, None)


ATTRIBUTE_DISPLAY_RANGES: Dict[str, Tuple[float | None, float | None]] = {
    "depth": DEPTH_DISPLAY_RANGE,
    "velocity": VELOCITY_DISPLAY_RANGE,
    "velocity_compensated_wrapped": VELOCITY_DISPLAY_RANGE,
    "velocity_measured_relative": VELOCITY_MEASURED_RELATIVE_DISPLAY_RANGE,
    "velocity_expected_static": VELOCITY_EXPECTED_STATIC_DISPLAY_RANGE,
    "velocity_compensated_pre_wrap": VELOCITY_COMPENSATED_PRE_WRAP_DISPLAY_RANGE,
    "ego_longitudinal_velocity": EGO_LONGITUDINAL_VELOCITY_DISPLAY_RANGE,
    "ego_yaw_rate": EGO_YAW_RATE_DISPLAY_RANGE,
    "rcs": RCS_DISPLAY_RANGE,
    "range_gate_length": RANGE_GATE_LENGTH_DISPLAY_RANGE,
    "doppler_range": DOPPLER_RANGE_DISPLAY_RANGE,
}
