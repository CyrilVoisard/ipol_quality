# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .io import events_table

#: Pre-processing of the angular velocity, as in the IPOL U-turn code (Butterworth, order, Hz)
FILTER_ORDER, FILTER_FC = 8, 14.0


@dataclass
class UTurn:
    """U-turn model M(i) = a E((i - b) / c) + C i on the common time base (indices in samples)."""
    start: int
    end: int
    amplitude: float          # |a| (deg)
    r2: float                 # coefficient of determination of the fit
    n_uturns: int             # number of U-turns counted (1 expected)
    a: float
    b: float
    c: float
    C: float                  # residual drift (deg per sample)
    fit_window: tuple
    drift_dps: float = 0.0    # drift removed before the fit (deg/s)
    q_low: float = 0.15       # quantiles of the centring of the heading
    q_high: float = 0.85
    index_shift: int = 0      # common grid index - index in the lower-back file


@dataclass
class Segmentation:
    uturn: Optional[UTurn] = None
    uturn_error: Optional[str] = None
    gait_events: Optional[pd.DataFrame] = None     # Foot (0 left, 1 right), TO, HS on the common grid
    gait_events_error: Optional[str] = None
    source: str = ""


# ---------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------

def _uturn_from_ipol(d, aligned):
    """UTurn from the content of the uturn.json file of the IPOL U-turn demo."""
    m = d["ModelParameters"]
    to_grid = lambda i: aligned.to_grid("LB", i)
    shift = to_grid(0)                      # b is a (real) sample index of the LB file
    params = d.get("Parameters", {})
    return UTurn(start=to_grid(d["UturnBoundaries"][0]), end=to_grid(d["UturnBoundaries"][1]),
                 amplitude=float(d["UturnAmplitude_deg"]), r2=float(d["Indicators"]["R2"]),
                 n_uturns=int(d["Indicators"]["UTurns"]), a=float(m["a_deg"]), b=float(m["b_sample"]) + shift,
                 c=float(m["c_samples"]), C=float(m["C_deg_per_sample"]),
                 fit_window=tuple(to_grid(i) for i in d["FitWindow"]),
                 drift_dps=float(d["Indicators"].get("Drift_dps", 0.0)),
                 q_low=float(params.get("Q_LOW", 0.15)), q_high=float(params.get("Q_HIGH", 0.85)),
                 index_shift=shift)


def _events_from_ipol(d, aligned):
    """Gait events (common grid) from the content of the gait_events.json of the MP-DTW demo."""
    table = events_table(d.get("LeftFootEvents"), d.get("RightFootEvents"))
    if table is None:
        return None
    for foot, sensor in ((0, "LF"), (1, "RF")):
        rows = table.Foot == foot
        for col in ("TO", "HS"):
            table.loc[rows, col] = [aligned.to_grid(sensor, i) for i in table.loc[rows, col]]
    return table.sort_values("HS").reset_index(drop=True)


def read_segmentation(source, aligned):
    """Segmentation from a segmentation file (path or already loaded dictionary)."""
    if isinstance(source, dict):
        d, name = source, "segmentation file"
    else:
        with open(source, "r") as file_id:
            d, name = json.load(file_id), str(source)
    seg = Segmentation(source=name)
    if not isinstance(d, dict):
        raise ValueError("the segmentation file must contain a JSON object")
    # the result file of one of the two IPOL demos may also be given alone
    if "UturnBoundaries" in d:
        d = {"uturn": d, "gait_events_error": "only the U-turn result was given (no gait events)"}
    elif "LeftFootEvents" in d or "RightFootEvents" in d:
        d = {"gait_events": d, "uturn_error": "only the gait events were given (no U-turn result)"}

    if d.get("uturn"):
        try:
            seg.uturn = _uturn_from_ipol(d["uturn"], aligned)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            seg.uturn_error = "unreadable U-turn result ({}: {})".format(type(error).__name__, error)
    else:
        seg.uturn_error = d.get("uturn_error") or "no U-turn segmentation given"

    if d.get("gait_events"):
        try:
            seg.gait_events = _events_from_ipol(d["gait_events"], aligned)
            if seg.gait_events is None:
                seg.gait_events_error = "no gait events in the segmentation file"
        except (KeyError, TypeError, ValueError) as error:
            seg.gait_events_error = "unreadable gait events ({}: {})".format(type(error).__name__, error)
    else:
        seg.gait_events_error = d.get("gait_events_error") or "no gait events given"
    return seg


def from_metadata(aligned):
    """Gait events published in the metadata of the public data sets (no U-turn model)."""
    meta = aligned.meta
    seg = Segmentation(source="metadata", uturn_error="no U-turn model in the metadata")
    if meta is None or meta.gait_events is None:
        seg.gait_events_error = "no gait events in the metadata"
    else:
        seg.gait_events = meta.gait_events.copy()
    return seg


# ---------------------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------------------

def low_pass(x, freq, order=FILTER_ORDER, fc=FILTER_FC):
    """Zero-phase Butterworth low-pass filter along the first axis."""
    x = np.asarray(x, dtype=float)
    if fc >= freq / 2.0:
        return x
    b, a = butter(order, fc / (freq / 2.0), btype="low")
    return filtfilt(b, a, x, axis=0)


def vertical_yaw_rate(aligned, sensor, rest=None):
    """Angular velocity about the vertical (deg/s): projection of the angular velocity on the
    direction of the gravity measured at rest (mean acceleration of ``rest``, a slice). This
    does not depend on how the sensor is mounted."""
    acc, gyr = aligned.acc[sensor], aligned.gyr[sensor]
    g = np.mean(acc[rest if rest is not None else slice(0, int(aligned.freq))], axis=0)
    return np.degrees(low_pass(gyr, aligned.freq) @ (g / np.linalg.norm(g)))


def lb_heading(aligned, uturn, axis=None):
    """Heading of the lower back as fitted by the U-turn code (for the figures).

    Integrated angular velocity about X (filtered, deg), corrected for the removed drift,
    centred between its two levels on the fit window and oriented so that it increases.
    ``axis``: X of the corrected file, on which the U-turn is computed (default: X of the sensor).

    Returns
    -------
    theta {array}, model on the fit window {array}
    """
    gyr = aligned.gyr["LB"][:, 0] if axis is None else aligned.gyr["LB"] @ np.asarray(axis, float)
    yaw = np.degrees(low_pass(gyr, aligned.freq))
    i = np.arange(aligned.n)
    theta = np.cumsum(yaw) / aligned.freq
    theta = theta - theta[0] - uturn.drift_dps / aligned.freq * i
    w0, w1 = max(uturn.fit_window[0], 0), min(uturn.fit_window[1], aligned.n)
    ref = theta[w0:w1]
    theta = theta - (np.quantile(ref, uturn.q_low) + np.quantile(ref, uturn.q_high)) / 2.0
    b = int(np.clip(round(uturn.b), w0 + 1, w1 - 1))
    if np.mean(theta[b:w1]) < np.mean(theta[w0:b]):
        theta = -theta
    x = np.arange(w0, w1, dtype=float)
    u = (x - uturn.b) / uturn.c
    step = np.where(u <= -0.5, -0.5, np.where(u >= 0.5, 0.5, 0.5 * np.sin(np.pi * u)))
    return theta, uturn.a * step + uturn.C * (x - uturn.index_shift)
