# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

from ..params import GRAVITY, SENSORS
from ..stillness import activity, quietest
from .base import absent, impossible, make

GROUP = "sensor"
_NAMES = (("still_acc", "m/s^2"), ("rest_acc", "%"), ("rest_gyr", "%"), ("gyro_bias", "deg/s"))


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
    where = ("quietest 1 s standing {} the walk".format(rest.label) if rest.label != "quietest"
             else "quietest 1 s of the recording (no standing period)")
    note = lambda text: "{}, {}".format(text, where)    # noqa: E731
    walk = active.end - active.start >= 2
    for s in SENSORS:
        if s not in aligned.acc:
            out += [absent(name, GROUP, s, unit) for name, unit in _NAMES]
            continue
        acc, gyr = aligned.acc[s], aligned.gyr[s]
        w, a = rest_window(aligned, stillness, s, st), active.slice()
        acc_norm = np.mean(np.linalg.norm(acc[w], axis=1))
        out.append(make("still_acc", GROUP, s, acc_norm, "m/s^2", abs(acc_norm - GRAVITY), params,
                        note=note("|a| = {:.2f} m/s^2".format(acc_norm))))
        abar = np.mean(acc[w], axis=0)
        dyn = np.linalg.norm(acc - abar, axis=1)
        gyr_norm = np.linalg.norm(gyr, axis=1)
        if walk and np.mean(dyn[a]) > 0 and np.mean(gyr_norm[a]) > 0:
            rest_acc = 100.0 * np.mean(dyn[w]) / np.mean(dyn[a])
            rest_gyr = 100.0 * np.mean(gyr_norm[w]) / np.mean(gyr_norm[a])
            out.append(make("rest_acc", GROUP, s, rest_acc, "%", rest_acc, params,
                            note=note("{:.0f} % of the walking level".format(rest_acc))))
            out.append(make("rest_gyr", GROUP, s, rest_gyr, "%", rest_gyr, params,
                            note=note("{:.0f} % of the walking level".format(rest_gyr))))
        else:
            out.append(impossible("rest_acc", GROUP, s, "%", "no walk detected"))
            out.append(impossible("rest_gyr", GROUP, s, "%", "no walk detected"))
        bias = np.rad2deg(np.linalg.norm(np.mean(gyr[w], axis=0)))
        out.append(make("gyro_bias", GROUP, s, bias, "deg/s", bias, params,
                        note=note("offset {:.1f} deg/s".format(bias))))
    return out
