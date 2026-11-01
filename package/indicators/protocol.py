# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

from ..params import FEET, SENSORS
from .base import absent, impossible, make, not_assessed
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
            out.append(not_assessed("orient_stability", GROUP, s, "deg",
                                    "no standing period {} the walk (scored by still_time)".format(
                                        " and ".join(missing))))
        else:
            g0 = np.mean(acc[rest_window(aligned, stillness, s, st, stillness.before)], axis=0)
            g1 = np.mean(acc[rest_window(aligned, stillness, s, st, stillness.after)], axis=0)
            cos = np.dot(g0, g1) / (np.linalg.norm(g0) * np.linalg.norm(g1))
            angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
            out.append(make("orient_stability", GROUP, s, angle, "deg", angle, params))
    return out
