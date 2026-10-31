# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Group "protocol": respect of the protocol and placement of the sensors.

Trial level:

* n_uturns   -- number of U-turns counted by the U-turn segmentation (1 expected):
                delta = |n - 1|; quality 0 if the U-turn is not segmented;
* still_time -- standing durations before (t_b, target 6 s) and after (t_a, target 2 s) the
                walk, capped at their targets: delta = 20 - (3 t_b + t_a) s.

Per sensor (the rest values are taken on the quietest second of the sensor, see sensor.py):

* orient_rest      -- share of the mean acceleration vector at rest (gravity) carried by the
                      expected axes (X for the head and the trunk, X and Z for the feet),
                      delta = 1 - share; quality 0 when the gravity has the wrong sign on one of
                      these axes (sensor upside down);
* walk_plane_acc   -- during the walk, share of the energy of the dynamic acceleration lying off
                      the sagittal plane (along Y);
* walk_plane_gyr   -- feet only: 1 - share of the energy of the angular velocity about Y;
* orient_stability -- angle between the mean acceleration vectors (tilt of the sensor) of the
                      standing periods before and after the walk: the sensor must not move on
                      the body during the trial (the heading change of the U-turn does not affect
                      it). Quality 0 when one of the two standing periods is missing: the
                      stability cannot be verified because the protocol was not followed.
"""

import numpy as np

from ..params import FEET, SENSORS
from .base import absent, impossible, make
from .sensor import rest_window

GROUP = "protocol"
_AXIS = {"X": 0, "Y": 1, "Z": 2}


def compute(aligned, stillness, segmentation, params):
    st = params["settings"]
    out = []

    # --- trial level ---------------------------------------------------------------------
    u = segmentation.uturn
    if u is None:
        out.append(impossible("n_uturns", GROUP, None, "-", "U-turn not segmented ({})".format(segmentation.uturn_error)))
    else:
        out.append(make("n_uturns", GROUP, None, u.n_uturns, "-", abs(u.n_uturns - 1), params))

    tb = min(stillness.t_before, st["still_before_target"])
    ta = min(stillness.t_after, st["still_after_target"])
    wb, wa = st["still_time_weights"]
    target = wb * st["still_before_target"] + wa * st["still_after_target"]
    out.append(make("still_time", GROUP, None, wb * tb + wa * ta, "s", max(0.0, target - wb * tb - wa * ta), params,
                    note="standing {:.1f} s before (6 s expected), {:.1f} s after (2 s expected)".format(
                        stillness.t_before, stillness.t_after)))

    # --- per sensor ----------------------------------------------------------------------------
    active = stillness.active
    walk = active.end - active.start >= 2
    off_plane = [_AXIS[k] for k in "XYZ" if k not in st["sagittal_axes"]]
    for s in SENSORS:
        if s not in aligned.acc:
            out += [absent("orient_rest", GROUP, s, "-"), absent("walk_plane_acc", GROUP, s, "-"),
                    absent("orient_stability", GROUP, s, "deg")]
            continue
        acc, gyr = aligned.acc[s], aligned.gyr[s]
        w = rest_window(aligned, stillness, s, st)
        abar = np.mean(acc[w], axis=0)
        axes = st["rest_axes"][s]
        share = sum(abar[_AXIS[k]] ** 2 for k in axes) / np.sum(abar ** 2)
        gate = 1.0 if all(abar[_AXIS[k]] > 0 for k in axes) else 0.0
        main = int(np.argmax(np.abs(abar)))
        main_axis = "{}{}".format("+" if abar[main] > 0 else "-", "XYZ"[main])
        if not gate:
            note = "gravity along -{} instead of +{}: sensor upside down or axes swapped".format(
                "/".join(k for k in axes if abar[_AXIS[k]] <= 0), "/".join(axes))
        elif share < 0.75:
            note = "gravity mostly along {} ({:.0f} % on {}): sensor rotated or axes swapped".format(
                main_axis, 100 * share, "+".join(axes))
        else:
            note = ""
        if stillness.rest.label == "quietest":
            note = (note + "; " if note else "") + "no standing period: quietest second used"
        out.append(make("orient_rest", GROUP, s, share, "-", 1.0 - share, params, gate=gate, note=note))

        if walk:
            dyn = acc[active.slice()] - abar
            energy = np.mean(dyn ** 2, axis=0)
            off = float(np.sum(energy[off_plane]) / np.sum(energy))
            out.append(make("walk_plane_acc", GROUP, s, off, "-", off, params))
            if s in FEET:
                g = np.mean(gyr[active.slice()] ** 2, axis=0)
                share_y = float(g[_AXIS[st["foot_gyr_axis"]]] / np.sum(g))
                out.append(make("walk_plane_gyr", GROUP, s, share_y, "-", 1.0 - share_y, params))
        else:
            out.append(impossible("walk_plane_acc", GROUP, s, "-", "no walk detected"))
            if s in FEET:
                out.append(impossible("walk_plane_gyr", GROUP, s, "-", "no walk detected"))

        missing = [lab for lab, win in (("before", stillness.before), ("after", stillness.after)) if win is None]
        if missing:
            out.append(impossible("orient_stability", GROUP, s, "deg",
                                  "no standing period {} the walk".format(" and ".join(missing))))
        else:
            g0 = np.mean(acc[rest_window(aligned, stillness, s, st, stillness.before)], axis=0)
            g1 = np.mean(acc[rest_window(aligned, stillness, s, st, stillness.after)], axis=0)
            cos = np.dot(g0, g1) / (np.linalg.norm(g0) * np.linalg.norm(g1))
            angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
            out.append(make("orient_stability", GROUP, s, angle, "deg", angle, params))
    return out
