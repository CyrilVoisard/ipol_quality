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
    "missing": 6.0,             # [%] share of lost samples
    "max_gap": 0.5,             # [s] longest run of lost samples
    "disconnection": 4.0,       # [%] share of the recording not covered by the sensor
    # --- sensor: behaviour at rest (per sensor) ----------------------------------------
    "still_acc": 1.0,           # [m/s^2] | mean norm of the acceleration - g |
    "rest_acc": 40.0,           # [%] dynamic acceleration at rest / during the walk
    "rest_gyr": 40.0,           # [%] angular velocity at rest, offset removed / during the walk
    "gyro_bias": 9.0,           # [deg/s] norm of the mean angular velocity at rest (trunk, feet)
    # --- orientation: alignment of each sensor with its segment (trunk, feet) -----------
    # only gross errors: a tilt or a sway that the gait itself explains is not a defect
    "orient_rest": 0.8,         # [-] 1 - share of gravity on the expected axes at rest (sin^2 of the tilt)
    "walk_plane_acc": 0.9,      # [-] share of the dynamic acceleration off the sagittal plane
    "walk_plane_gyr": 0.9,      # [-] 1 - share of the foot angular velocity about Y
    "orient_stability": 15.0,   # [deg] tilt change of the sensor between start and end
    # --- protocol (trial level) ------------------------------------------------------------
    "still_time": 2.0,          # [s] shortfall of the standing periods (see indicators/protocol.py)
    # --- U-turn ------------------------------------------------------------------------------
    "uturn_fit": 0.1,           # [-] 1 - R^2 of the turning model (75 at R^2 = 0.95, 50 at 0.9)
    "uturn_amplitude": 45.0,    # [deg] | 180 - U-turn amplitude |
    "drift_ratio": 0.2,         # [-] offset of the trunk gyroscope about the vertical / mean turning rate
    "head_trunk": 0.3,          # [-] 1 - ratio of the head and trunk turn amplitudes
    # --- segmentation ------------------------------------------------------------------------
    "alternation": 10.0,        # [%] 100 - right/left alternation of the gait events
}

#: Tolerance: deviation up to which the quality stays 100 (0 when not listed).
DELTA100 = {
    "gyro_bias": 3.0,           # an offset this small is within the specifications of the sensors
    "walk_plane_acc": 0.5,      # half of the energy of the walk in the expected plane is enough
    "walk_plane_gyr": 0.5,
    "drift_ratio": 0.025,       # 4.5 deg added over a U-turn of 180 deg
}

#: Scale of the mapping ("linear" when not listed): "log" for a deviation that spans orders of
#: magnitude (each doubling costs 25 points: 100 up to 0.025, 75 at 0.07, 50 at 0.2 for the drift).
SCALE = {"drift_ratio": "log"}

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
    # quality of the number of U-turns counted: 1, 2, 3 (0 beyond)
    "n_uturns_quality": (100.0, 75.0, 50.0),
    # True for the analyses that need the end of the walk (shape of the last steps, gait termination):
    # a recording whose sensors still move at its end then gets still_time = 0
    "end_standing_required": False,
    # stillness detection
    # the head may move while standing: it is left out of the detection of the standing periods,
    # of the indicators of movement and offset at rest (rest_acc, rest_gyr, gyro_bias) and of the
    # orientation group (its tilt and its sway are behaviour; its placement is still corrected)
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
    # classes of a quality: good above "good", acceptable above "critical" up to "good", critical
    # up to "critical"; a quality on a boundary falls in the more severe class
    # (the per-sensor, group and global scores are the minimum of what they contain: scoring.py)
    "critical": 50.0,
    "good": 75.0,
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
