# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parameters of the quality score.

Every indicator is mapped to [0, 100] by the same bounded linear function of its
deviation delta >= 0 from the ideal value (delta = 0):

    Q(delta; delta50) = 100 * max(0, 1 - delta / (2 * delta50)),

so that Q(0) = 100, Q(delta50) = 50 and Q(2 * delta50) = 0. The only parameter of an
indicator is therefore its half-quality deviation delta50, gathered in ``DELTA50``.
The values below are the a priori values of the first version of the score; they are
meant to be revised by the calibration study.

The other parameters (``SETTINGS``) are the constants of the computation itself
(sensor axes, protocol durations, stillness detection).
"""

import copy

GRAVITY = 9.81  # m/s^2

SENSORS = ("HE", "LB", "LF", "RF")   # head, lower back, left foot, right foot
FEET = ("LF", "RF")
REQUIRED_SENSORS = ("LB", "LF", "RF")  # the head sensor is optional

#: Half-quality deviation of each indicator (unit of the deviation in brackets).
DELTA50 = {
    # --- raw: integrity of the data stream (per sensor) ----------------------------
    "missing": 50.0,            # [%] share of lost samples
    "max_gap": 1.0,             # [s] longest run of lost samples
    "disconnection": 10.0,      # [%] share of the recording not covered by the sensor
    # --- sensor: behaviour at rest (per sensor) ----------------------------------------
    "still_acc": 0.25 * GRAVITY,  # [m/s^2] | mean norm of the acceleration - g |
    "rest_acc": 25.0,           # [%] dynamic acceleration at rest / during the walk
    "rest_gyr": 25.0,           # [%] angular velocity at rest / during the walk
    "gyro_bias": 2.0,           # [deg/s] norm of the mean angular velocity at rest
    # --- protocol: protocol and sensor placement -------------------------------------
    "n_uturns": 2.0,            # [-] | number of U-turns - 1 |   (-25 per extra turn)
    "still_time": 10.0,         # [s] shortfall of 3 * t_before + t_after w.r.t. 20 s
    "orient_rest": 0.5,         # [-] 1 - share of gravity on the expected axes at rest
    "walk_plane_acc": 0.5,      # [-] share of the dynamic acceleration off the sagittal plane
    "walk_plane_gyr": 0.5,      # [-] 1 - share of the foot angular velocity about Y
    "orient_stability": 20.0,   # [deg] tilt change of the sensor between start and end
    # --- U-turn ------------------------------------------------------------------------------
    "uturn_fit": 0.25,          # [-] 1 - R^2 of the turning model
    "uturn_amplitude": 90.0,    # [deg] | 180 - U-turn amplitude |
    "straight_dev": 0.5,        # [-] residual drift of the U-turn model, |C| c / |a|
    "head_trunk": 0.5,          # [-] 1 - ratio of the head and trunk turn amplitudes
    # --- segmentation ------------------------------------------------------------------------
    "alternation": 25.0,        # [%] 100 - right/left alternation of the gait events
}

SETTINGS = {
    # expected axes of the gravity at rest, with a positive sign (sensor frames of the
    # data sets: X along the cranio-caudal axis for the head and the trunk, foot flat
    # with the gravity shared between X and Z)
    "rest_axes": {"HE": ("X",), "LB": ("X",), "LF": ("X", "Z"), "RF": ("X", "Z")},
    # axes of the sagittal plane, and axis of the main angular velocity of the feet
    "sagittal_axes": ("X", "Z"),
    "foot_gyr_axis": "Y",
    # protocol: standing durations required before and after the walk (s)
    "still_before_target": 6.0,
    "still_after_target": 2.0,
    "still_time_weights": (3.0, 1.0),
    # stillness detection
    "still_sensors": ("LB", "LF", "RF"),   # the head may move while standing
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
}


def default_params():
    """Deep copy of the default parameters, to be modified freely."""
    return {"delta50": copy.deepcopy(DELTA50), "settings": copy.deepcopy(SETTINGS)}


def quality(delta, delta50):
    """Bounded linear quality Q(delta; delta50) in [0, 100] (None if delta is None)."""
    if delta is None:
        return None
    return 100.0 * max(0.0, 1.0 - float(delta) / (2.0 * float(delta50)))


def load_params(path=None):
    """Default parameters, updated with a JSON file {"delta50": {...}, "settings": {...}} if given
    (e.g. the half-quality deviations proposed by the calibration)."""
    import json
    params = default_params()
    if path:
        with open(path, "r") as f:
            custom = json.load(f)
        for key in ("delta50", "settings"):
            unknown = set(custom.get(key, {})) - set(params[key])
            if unknown:
                raise ValueError("unknown {} parameters: {}".format(key, sorted(unknown)))
            params[key].update(custom.get(key, {}))
    return params
