# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

from ..params import FEET, SENSORS
from .base import absent, impossible, make, not_assessed
from .sensor import rest_window

GROUP = "protocol"          # trial level: number of U-turns, standing time
ORIENTATION = "orientation"     # per sensor: purity of the orientation, plane of the walk, stability
_AXIS = {"X": 0, "Y": 1, "Z": 2}


def compute(aligned, stillness, segmentation, params):
    st = params["settings"]
    out = []

    # --- trial level ---------------------------------------------------------------------
    u = segmentation.uturn
    if u is None:
        out.append(impossible("n_uturns", GROUP, None, "-", "U-turn not segmented ({})".format(segmentation.uturn_error)))
    else:
        extra = abs(u.n_uturns - 1)
        table = st["n_uturns_quality"]
        out.append(make("n_uturns", GROUP, None, u.n_uturns, "-", extra, params,
                        q=table[extra] if extra < len(table) else 0.0,
                        note="{} U-turn{} counted".format(u.n_uturns, "s" if u.n_uturns != 1 else "")))

    # shortfall of the standing periods: the larger of the shortfall before the walk and the mean
    # shortfall of the two periods. With 4 s / 2 s and delta50 = 2 s: 4 s and 2 s give 100, 3 s and
    # 1 s give 75, and less than 2 s before the walk gives less than 50 whatever comes after
    tb_target, ta_target = st["still_before_target"], st["still_after_target"]
    short_b = tb_target - min(stillness.t_before, tb_target)
    short_a = ta_target - min(stillness.t_after, ta_target)
    note = "standing {:.1f} s before ({:g} s expected), {:.1f} s after ({:g} s expected)".format(
        stillness.t_before, tb_target, stillness.t_after, ta_target)
    # end of the walk recorded: the sensors are still at the end of the recording
    ended = stillness.t_after > 0
    if not ended:
        note += "; sensors still moving at the end: the end of the walk is not recorded"
    out.append(make("still_time", GROUP, None, stillness.t_before + stillness.t_after, "s",
                    max(short_b, (short_b + short_a) / 2.0), params,
                    gate=0.0 if st["end_standing_required"] and not ended else 1.0, note=note))

    # --- per sensor ----------------------------------------------------------------------------
    active = stillness.active
    walk = active.end - active.start >= 2
    off_plane = [_AXIS[k] for k in "XYZ" if k not in st["sagittal_axes"]]
    for s in SENSORS:
        if s not in st["still_sensors"]:
            continue    # the head: its tilt and its sway are behaviour, not a defect of the recording
        if s not in aligned.acc:
            out += [absent("orient_rest", ORIENTATION, s, "-"), absent("walk_plane_acc", ORIENTATION, s, "-"),
                    absent("orient_stability", ORIENTATION, s, "deg")]
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
        elif share < 0.75 and s in st.get("tilt_sensors", ()):
            # tilt kept as recorded in the score; the corrected file is aligned on the vertical
            note = "tilted {:.0f}° from the vertical at rest ({:.0f} % of the gravity on X): " \
                   "X aligned on the vertical in corrected_{}.txt".format(
                       np.degrees(np.arccos(np.sqrt(share))), 100 * share, s)
        elif share < 0.75:
            note = "gravity mostly along {} ({:.0f} % on {}): sensor rotated or axes swapped".format(
                main_axis, 100 * share, "+".join(axes))
        else:
            note = ""
        if stillness.rest.label == "quietest":
            note = (note + "; " if note else "") + "no standing period: quietest second used"
        out.append(make("orient_rest", ORIENTATION, s, share, "-", 1.0 - share, params, gate=gate, note=note))

        if walk:
            dyn = acc[active.slice()] - abar
            energy = np.mean(dyn ** 2, axis=0)
            off = float(np.sum(energy[off_plane]) / np.sum(energy))
            out.append(make("walk_plane_acc", ORIENTATION, s, off, "-", off, params))
            if s in FEET:
                g = np.mean(gyr[active.slice()] ** 2, axis=0)
                share_y = float(g[_AXIS[st["foot_gyr_axis"]]] / np.sum(g))
                out.append(make("walk_plane_gyr", ORIENTATION, s, share_y, "-", 1.0 - share_y, params))
        else:
            out.append(impossible("walk_plane_acc", ORIENTATION, s, "-", "no walk detected"))
            if s in FEET:
                out.append(impossible("walk_plane_gyr", ORIENTATION, s, "-", "no walk detected"))

        missing = [lab for lab, win in (("before", stillness.before), ("after", stillness.after)) if win is None]
        if missing:
            out.append(not_assessed("orient_stability", ORIENTATION, s, "deg",
                                    "no standing period {} the walk (scored by still_time)".format(
                                        " and ".join(missing))))
        else:
            g0 = np.mean(acc[rest_window(aligned, stillness, s, st, stillness.before)], axis=0)
            g1 = np.mean(acc[rest_window(aligned, stillness, s, st, stillness.after)], axis=0)
            cos = np.dot(g0, g1) / (np.linalg.norm(g0) * np.linalg.norm(g1))
            angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
            out.append(make("orient_stability", ORIENTATION, s, angle, "deg", angle, params))
    return out
