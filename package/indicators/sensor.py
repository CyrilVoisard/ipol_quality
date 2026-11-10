# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

from ..params import GRAVITY, SENSORS
from ..stillness import activity, quietest
from .base import absent, impossible, make, not_assessed

GROUP = "sensor"
_NAMES = (("still_acc", "m/s^2"), ("rest_acc", "%"), ("rest_gyr", "%"), ("gyro_bias", "deg/s"))
#: indicators of movement and offset at rest, not computed for the head (it may move while standing)
_BODY_ONLY = ("rest_acc", "rest_gyr", "gyro_bias")


def rest_window(aligned, stillness, sensor, settings, window=None):
    """Quietest sub-window of ``sensor`` within ``window`` (default: the rest window)."""
    window = window or stillness.rest
    _, act = activity(aligned, (sensor,), settings["still_win"])
    length = int(round(settings["rest_sub"] * aligned.freq))
    return slice(*quietest(act, length, window.start, window.end))


def compute(aligned, stillness, segmentation, params):
    st = params["settings"]
    out = []
    rest, active = stillness.rest, stillness.active
    where = "quietest 1 s standing {} the walk".format(rest.label)
    note = lambda text: "{}, {}".format(text, where)    # noqa: E731
    walk = active.end - active.start >= 2
    for s in SENSORS:
        names = [(n, u) for n, u in _NAMES if s in st["still_sensors"] or n not in _BODY_ONLY]
        if s not in aligned.acc:
            out += [absent(name, GROUP, s, unit) for name, unit in names]
            continue
        if rest.label == "quietest":
            # no standing period at all: the quietest second is taken while walking, so the values at
            # rest would score the walk instead of the sensor; the missing standing periods are scored
            # by still_time
            out += [not_assessed(name, GROUP, s, unit, "no standing period (scored by still_time)")
                    for name, unit in names]
            continue
        acc, gyr = aligned.acc[s], aligned.gyr[s]
        w, a = rest_window(aligned, stillness, s, st), active.slice()
        acc_norm = np.mean(np.linalg.norm(acc[w], axis=1))
        out.append(make("still_acc", GROUP, s, acc_norm, "m/s^2", abs(acc_norm - GRAVITY), params,
                        note=note("|a| = {:.2f} m/s^2".format(acc_norm))))
        if s not in st["still_sensors"]:
            continue    # the head may move while the subject stands still: no indicator of movement at rest
        abar = np.mean(acc[w], axis=0)
        dyn = np.linalg.norm(acc - abar, axis=1)
        if walk:
            # movement at rest relative to the walk; the angular velocity without its offset, which
            # is scored by gyro_bias alone
            gyr_rest = np.linalg.norm(gyr[w] - np.mean(gyr[w], axis=0), axis=1)
            gyr_walk = np.linalg.norm(gyr[a] - np.mean(gyr[a], axis=0), axis=1)
            if np.mean(dyn[a]) > 0 and np.mean(gyr_walk) > 0:
                rest_acc = 100.0 * np.mean(dyn[w]) / np.mean(dyn[a])
                rest_gyr = 100.0 * np.mean(gyr_rest) / np.mean(gyr_walk)
                out.append(make("rest_acc", GROUP, s, rest_acc, "%", rest_acc, params,
                                note=note("{:.0f} % of the walking level".format(rest_acc))))
                out.append(make("rest_gyr", GROUP, s, rest_gyr, "%", rest_gyr, params,
                                note=note("{:.0f} % of the walking level".format(rest_gyr))))
            else:
                out.append(impossible("rest_acc", GROUP, s, "%", "no movement during the walk"))
                out.append(impossible("rest_gyr", GROUP, s, "%", "no movement during the walk"))
        else:
            out.append(impossible("rest_acc", GROUP, s, "%", "no walk detected"))
            out.append(impossible("rest_gyr", GROUP, s, "%", "no walk detected"))
        bias = np.rad2deg(np.linalg.norm(np.mean(gyr[w], axis=0)))
        out.append(make("gyro_bias", GROUP, s, bias, "deg/s", bias, params,
                        note=note("offset {:.1f} deg/s".format(bias))))
    return out
