# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import os

import numpy as np
import pandas as pd

from .params import GRAVITY, SENSORS

_REST_GRAVITY = {"HE": (9.70, 0.40, 1.30), "LB": (9.78, 0.15, 0.70), "LF": (5.0, 0.0, 8.43), "RF": (5.0, 0.0, 8.43)}


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v) * GRAVITY


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def simulate(freq=100, t_before=4.0, t_walk=8.0, t_turn=2.0, t_after=2.0, stride=1.1,
             gyro_bias=None, noise=True, seed=0):
    """Simulate the four sensors (aligned, SI units).

    Returns
    -------
    signals {dict sensor -> (acc (n,3), gyr (n,3))}, truth {dict}
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / freq
    t_end = t_before + 2 * t_walk + t_turn + t_after
    n = int(round(t_end * freq))
    t = np.arange(n) * dt
    walk1 = (t >= t_before) & (t < t_before + t_walk)
    turn = (t >= t_before + t_walk) & (t < t_before + t_walk + t_turn)
    walk2 = (t >= t_before + t_walk + t_turn) & (t < t_before + 2 * t_walk + t_turn)
    walking = walk1 | walk2 | turn

    # --- yaw rate of the U-turn (rad/s), integral = pi -----------------------------------
    u = (t - t_before - t_walk) / t_turn
    yaw = np.where(turn, (np.pi ** 2 / (2 * t_turn)) * np.sin(np.pi * u), 0.0)

    signals, events = {}, []
    # --- feet ------------------------------------------------------------------------------
    swing = 0.4 * stride
    for k, s in enumerate(("LF", "RF")):
        gyr = np.zeros((n, 3))
        starts = np.arange(t_before + 0.2 + k * stride / 2, t_before + 2 * t_walk + t_turn - swing, stride)
        for t0 in starts:
            i0, i1 = int(round(t0 * freq)), int(round((t0 + swing) * freq))
            v = np.linspace(0, 1, i1 - i0)
            # sagittal rotation of the swing, zero mean: the foot is flat again at heel strike
            gyr[i0:i1, 1] += 5.0 * (np.sin(np.pi * v) - (4 / np.pi) * np.sin(2 * np.pi * v) ** 2)
            gyr[i0:i1, 0] += 0.6 * np.sin(2 * np.pi * v)
            gyr[i0:i1, 2] += 0.4 * np.sin(np.pi * v)
            events.append((k, i0, i1))
        pitch = np.cumsum(gyr[:, 1]) * dt
        g = np.array(_unit(_REST_GRAVITY[s]))
        acc = np.array([_rot_y(-p) @ g for p in pitch])
        dyn = np.zeros((n, 3))
        dyn[:, 0] = 6.0 * np.gradient(gyr[:, 1], dt) / 60.0
        dyn[:, 2] = 4.0 * gyr[:, 1] / 5.0
        dyn[:, 1] = 0.6 * gyr[:, 0]
        gyr[:, 0] += yaw * 0.9
        signals[s] = (acc + dyn, gyr)

    # --- trunk and head ----------------------------------------------------------------------
    step_f = 2.0 / stride
    for s in ("LB", "HE"):
        g = np.array(_unit(_REST_GRAVITY[s]))
        acc = np.tile(g, (n, 1))
        acc[:, 0] += np.where(walking, 2.0 * np.sin(2 * np.pi * step_f * t), 0.0)
        acc[:, 2] += np.where(walking, 1.2 * np.sin(2 * np.pi * step_f * t + 1.0), 0.0)
        acc[:, 1] += np.where(walking, 0.6 * np.sin(np.pi * step_f * t), 0.0)
        gyr = np.zeros((n, 3))
        lead = int(0.1 * freq) if s == "HE" else 0
        gyr[:, 0] = np.roll(yaw, -lead) + np.where(walking, 0.3 * np.sin(np.pi * step_f * t), 0.0)
        gyr[:, 1] = np.where(walking, 0.25 * np.sin(2 * np.pi * step_f * t), 0.0)
        gyr[:, 2] = np.where(walking, 0.2 * np.sin(np.pi * step_f * t + 0.5), 0.0)
        signals[s] = (acc, gyr)

    for s in SENSORS:
        acc, gyr = signals[s]
        if noise:
            acc = acc + rng.normal(0, 0.02, acc.shape)
            gyr = gyr + rng.normal(0, 0.004, gyr.shape)
        bias = np.zeros(3) if gyro_bias is None or s not in gyro_bias else np.asarray(gyro_bias[s], float)
        signals[s] = (acc, gyr + bias)

    ev = pd.DataFrame(events, columns=["Foot", "TO", "HS"]).sort_values("HS").reset_index(drop=True)
    i_turn = int(round((t_before + t_walk) * freq))
    truth = {"freq": freq, "n": n, "t_before": t_before, "t_after": t_after,
             "uturn": (i_turn, i_turn + int(round(t_turn * freq))), "gait_events": ev}
    return signals, truth


def segmentation_dict(truth, trim_start=None):
    """Segmentation file (IPOL demo formats) built from the ground truth of a synthetic trial.

    ``trim_start`` (sensor -> samples removed at the beginning of its file) shifts the indices to
    the frame of each file, as the IPOL demos count them."""
    trim_start = trim_start or {}
    n, (s0, s1) = truth["n"], truth["uturn"]
    lb = trim_start.get("LB", 0)
    b, c = (s0 + s1) / 2.0, float(s1 - s0)
    w0, w1 = int(np.ceil(0.4 * b)), int(b + np.floor(0.6 * (n - b)))
    uturn = {"Freq": truth["freq"], "UturnBoundaries": [s0 - lb, s1 - lb], "UturnAmplitude_deg": 180.0,
             "Indicators": {"R2": 0.998, "UTurns": 1, "Drift_dps": 0.0},
             "FitWindow": [w0 - lb, w1 - lb],
             "ModelParameters": {"a_deg": 180.0, "b_sample": b - lb, "c_samples": c, "C_deg_per_sample": 0.0},
             "Parameters": {"Q_LOW": 0.15, "Q_HIGH": 0.85}}
    ev = truth["gait_events"]
    pairs = lambda foot, s: [[int(r.TO) - trim_start.get(s, 0), int(r.HS) - trim_start.get(s, 0)]
                             for r in ev[ev.Foot == foot].itertuples()]
    return {"uturn": uturn, "gait_events": {"Freq": truth["freq"], "LeftFootEvents": pairs(0, "LF"),
                                            "RightFootEvents": pairs(1, "RF")}}


def write_trial(folder, trial_id="SYN_1_1", layout="2025", freq=100, counter_start=0,
                drop=None, truncate=None, trim_start=None, upside_down=(), tilt_after=None, head=True,
                **sim_kwargs):
    """Write a synthetic trial in the layout of a public data set.

    Arguments:
        folder {str} -- folder created for the trial
        layout {str} -- "2025" (flat metadata, magnetometer columns) or "amisia" (nested metadata)
        counter_start {int} -- first value of the packet counters (wraps at 65536)
        drop {dict} -- sensor -> list of (first sample, number of samples) removed from the file
        truncate {dict} -- sensor -> number of samples removed at the end of the file
        trim_start {dict} -- sensor -> number of samples removed at the beginning of the file
        head {bool} -- write the head sensor file
        upside_down {tuple} -- sensors turned by 180 deg about Y (X and Z reversed)
        tilt_after {dict} -- sensor -> tilt (deg, about Y) of the sensor after the walk
        sim_kwargs -- passed to :func:`simulate`

    A segmentation file ``<trial>_segmentation.json`` is written from the ground truth.

    Returns
    -------
    truth {dict} (with the path of the segmentation file under "segmentation")
    """
    signals, truth = simulate(freq=freq, **sim_kwargs)
    os.makedirs(folder, exist_ok=True)
    n = truth["n"]
    for s in SENSORS:
        if s == "HE" and not head:
            continue
        acc, gyr = (x.copy() for x in signals[s])
        if s in upside_down:
            acc[:, [0, 2]] *= -1
            gyr[:, [0, 2]] *= -1
        if tilt_after and s in tilt_after:
            i_after = n - int(round(truth["t_after"] * freq))
            r = _rot_y(np.radians(tilt_after[s]))
            acc[i_after:] = acc[i_after:] @ r.T
        counter = (counter_start + np.arange(n)) % 65536
        keep = np.ones(n, dtype=bool)
        for first, length in (drop or {}).get(s, []):
            keep[first:first + length] = False
        if truncate and s in truncate:
            keep[n - truncate[s]:] = False
        if trim_start and s in trim_start:
            keep[:trim_start[s]] = False
        cols = {"PacketCounter": counter}
        for k, a in enumerate("XYZ"):
            cols["Acc_" + a] = acc[:, k]
        for k, a in enumerate("XYZ"):
            cols["Gyr_" + a] = gyr[:, k]
        if layout == "2025":
            for a in "XYZ":
                cols["Mag_" + a] = 0.5
        pd.DataFrame(cols)[keep].to_csv(os.path.join(folder, "{}_raw_data_{}.txt".format(trial_id, s)),
                                        sep="\t", index=False, float_format="%.6f")

    ev = truth["gait_events"]
    left = ev[ev.Foot == 0][["TO", "HS"]].values.tolist()
    right = ev[ev.Foot == 1][["TO", "HS"]].values.tolist()
    if layout == "2025":
        meta = {"subject": trial_id.rsplit("_", 1)[0], "group": "healthy", "pathologyKey": "HS",
                "freq": float(freq), "sensor": "synthetic", "uturnBoundaries": list(truth["uturn"]),
                "leftGaitEvents": left, "rightGaitEvents": right}
    else:
        meta = {"demographics": {"participant_id": 1},
                "gait": {"sensor": "synthetic", "freq": freq, "uturn_boundaries_samples": list(truth["uturn"]),
                         "left_gait_events_samples": left, "right_gait_events_samples": right}}
    with open(os.path.join(folder, "{}_meta.json".format(trial_id)), "w") as f:
        json.dump(meta, f)
    truth["segmentation"] = os.path.join(folder, "{}_segmentation.json".format(trial_id))
    with open(truth["segmentation"], "w") as f:
        json.dump(segmentation_dict(truth, trim_start), f)
    return truth
