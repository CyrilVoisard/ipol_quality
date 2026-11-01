# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy

import numpy as np

GRAVITY = 9.81  # m/s^2

SENSORS = ("HE", "LB", "LF", "RF")   # head, lower back, left foot, right foot
FEET = ("LF", "RF")
REQUIRED_SENSORS = ("LB", "LF", "RF")  # the head sensor is optional

#: Half-quality deviation of each indicator (unit of the deviation in brackets).
DELTA50 = {
    # --- raw: integrity of the data stream (per sensor) ----------------------------
    "missing": 10.0,            # [%] share of lost samples
    "max_gap": 0.5,             # [s] longest run of lost samples
    "disconnection": 4.0,       # [%] share of the recording not covered by the sensor
    # --- sensor: behaviour at rest (per sensor) ----------------------------------------
    "still_acc": 0.5,           # [m/s^2] | mean norm of the acceleration - g |
    "rest_acc": 10.0,           # [%] dynamic acceleration at rest / during the walk
    "rest_gyr": 10.0,           # [%] angular velocity at rest, offset removed / during the walk
    "gyro_bias": 8.0,           # [deg/s] norm of the mean angular velocity at rest
    # --- orientation: alignment of each sensor with its segment (per sensor) -----------
    "orient_rest": 0.3,         # [-] 1 - share of gravity on the expected axes at rest
    "walk_plane_acc": 0.5,      # [-] share of the dynamic acceleration off the sagittal plane
    "walk_plane_gyr": 0.5,      # [-] 1 - share of the foot angular velocity about Y
    "orient_stability": 10.0,   # [deg] tilt change of the sensor between start and end
    # --- protocol (trial level) ------------------------------------------------------------
    "n_uturns": 2.0,            # [-] | number of U-turns - 1 |   (-25 per extra turn)
    "still_time": 2.0,          # [s] shortfall of the standing periods (see indicators/protocol.py)
    # --- U-turn ------------------------------------------------------------------------------
    "uturn_fit": 0.05,          # [-] 1 - R^2 of the turning model
    "uturn_amplitude": 35.0,    # [deg] | 180 - U-turn amplitude |
    "straight_dev": 0.5,        # [-] residual drift of the U-turn model, |C| c / |a|
    "head_trunk": 0.3,          # [-] 1 - ratio of the head and trunk turn amplitudes
    # --- segmentation ------------------------------------------------------------------------
    "alternation": 10.0,        # [%] 100 - right/left alternation of the gait events
}

#: Tolerance: deviation up to which the quality stays 100 (0 when not listed).
DELTA100 = {
    "gyro_bias": 2.0,           # an offset this small is within the specifications of the sensors
    "walk_plane_acc": 0.25,     # 75 % of the energy of the walk in the expected plane is enough
    "walk_plane_gyr": 0.25,
}

#: Scale of the mapping ("linear" when not listed): "log" for a deviation that spans orders of
#: magnitude (each doubling of the offset costs the same: 100 at 2, 75 at 4, 50 at 8 deg/s).
SCALE = {"gyro_bias": "log"}

SETTINGS = {
    # expected axes of the gravity at rest, with a positive sign (sensor frames of the
    # data sets: X along the cranio-caudal axis for the head and the trunk, foot flat
    # with the gravity shared between X and Z)
    "rest_axes": {"HE": ("X",), "LB": ("X",), "LF": ("X", "Z"), "RF": ("X", "Z")},
    # axes of the sagittal plane, and axis of the main angular velocity of the feet
    "sagittal_axes": ("X", "Z"),
    "foot_gyr_axis": "Y",
    # protocol: standing durations required before and after the walk (s)
    "still_before_target": 4.0,
    "still_after_target": 2.0,
    # stillness detection
    # the head may move while standing: it is left out of the detection of the standing periods
    # and of the indicators of movement at rest (rest_acc, rest_gyr) and of orientation stability
    "still_sensors": ("LB", "LF", "RF"),
    "still_win": 0.5,           # [s] window of the moving standard deviation
    "still_rel": 0.10,          # activity threshold, fraction of the walking level
    "still_floor_acc": 0.05,    # [m/s^2] lower bound of the acceleration threshold
    "still_floor_gyr": 0.05,    # [rad/s] lower bound of the angular velocity threshold
    "still_merge_gap": 1.0,     # [s] pauses shorter than this are part of the walk
    "still_margin": 0.3,        # [s] removed next to the walk
    "still_min_duration": 0.3,  # [s] minimal duration of a usable still window
    # sensor indicators: computed, for each sensor, on its quietest sub-window of the rest window
    "rest_sub": 1.0,            # [s] duration of that sub-window
    # U-turn: margin around the turn for the head/trunk comparison (s)
    "head_trunk_margin": 1.0,
    # aggregation: quality below which an element is critical, and rule of the group scores
    # ("critical_min": mean if every element reaches the critical quality, minimum otherwise;
    # "mean": plain mean)
    "critical": 50.0,
    "aggregation": "critical_min",
    # placement of the sensors: "detect" finds the rotation of each sensor in its plane and
    # corrects it before the indicators (reported, not scored); "fixed" assumes classic frames
    "placement": "detect",
}


def default_params():
    """Deep copy of the default parameters, to be modified freely."""
    return {"delta50": copy.deepcopy(DELTA50), "delta100": copy.deepcopy(DELTA100), "scale": copy.deepcopy(SCALE),
            "settings": copy.deepcopy(SETTINGS)}


def quality(delta, delta50, delta100=0.0, scale="linear"):
    """Quality in [0, 100] of a deviation (None if delta is None): 100 up to delta100, 50 at
    delta50, linear in delta (or in log delta) in between and beyond, 0 from
    2 delta50 - delta100 (or delta50^2 / delta100)."""
    if delta is None:
        return None
    d, d50, d100 = float(delta), float(delta50), float(delta100)
    if d <= d100:
        return 100.0
    if scale == "log":
        u = np.log(d / d100) / np.log(d50 / d100)
    else:
        u = (d - d100) / (d50 - d100)
    return 100.0 * max(0.0, 1.0 - u / 2.0)


def load_params(path=None):
    """Default parameters, updated with a JSON file {"delta50": {...}, "delta100": {...},
    "scale": {...}, "settings": {...}} if given."""
    import json
    params = default_params()
    if path:
        with open(path, "r") as f:
            custom = json.load(f)
        for key in ("delta50", "delta100", "scale", "settings"):
            unknown = set(custom.get(key, {})) - (set(params[key]) if key != "delta100" and key != "scale"
                                                  else set(params["delta50"]))
            if unknown:
                raise ValueError("unknown {} parameters: {}".format(key, sorted(unknown)))
            params[key].update(custom.get(key, {}))
    return params
