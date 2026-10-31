# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reading of the raw recordings.

A trial is made of one raw file per sensor -- lower back (LB), left foot (LF), right foot (RF)
and, optionally, head (HE) -- each holding the columns ``PacketCounter``, ``Acc_X``, ``Acc_Y``,
``Acc_Z``, ``Gyr_X``, ``Gyr_Y``, ``Gyr_Z`` (other columns are ignored, header lines before
``PacketCounter`` are skipped).

The files can be given explicitly (IPOL demo) or found in a trial folder of the public data
sets, which share the layout ``<trial>_raw_data_{HE,LB,LF,RF}.txt`` + ``<trial>_meta.json``:

* Voisard et al., Scientific Data 12:1674 (2025): flat metadata (``freq``, ``uturnBoundaries``,
  ``leftGaitEvents``...);
* AMISIA cohort (Limoges): metadata nested in categories (``gait.freq``...).

Units: accelerations in m/s^2 or g, angular velocities in rad/s or deg/s. They are given
(``acc_unit``, ``gyr_unit``) or recognised from the order of magnitude of the signal ("auto").
"""

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .params import GRAVITY, REQUIRED_SENSORS, SENSORS

_RAW_RE = re.compile(r"^(?P<trial>.+)_raw_data_(?P<sensor>HE|LB|LF|RF)\.txt$")
_AXES = ("X", "Y", "Z")
ACC_COLS = tuple("Acc_" + a for a in _AXES)
GYR_COLS = tuple("Gyr_" + a for a in _AXES)


@dataclass
class TrialMeta:
    """Fields of the metadata of the public data sets (``content`` keeps the whole file)."""
    freq: Optional[float] = None
    sensor_type: Optional[str] = None
    group: Optional[str] = None
    pathology: Optional[str] = None
    subject: Optional[str] = None
    uturn_boundaries: Optional[tuple] = None
    gait_events: Optional[pd.DataFrame] = None     # columns Foot (0 left, 1 right), TO, HS
    content: dict = field(default_factory=dict, repr=False)


@dataclass
class RawTrial:
    """Raw recordings of the sensors of a trial (SI units)."""
    trial_id: str
    freq: float
    sensors: dict                 # sensor -> DataFrame (PacketCounter, Acc_*, Gyr_*)
    meta: TrialMeta = field(default_factory=TrialMeta)
    trial_dir: Optional[str] = None
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------------------

def find_trial_files(trial_dir):
    """Raw files and metadata of the trial stored in ``trial_dir``.

    Returns
    -------
    trial_id {str}, raw {dict sensor -> path}, meta {str or None}
    """
    raw, trial_id = {}, None
    for path in sorted(glob.glob(os.path.join(trial_dir, "*_raw_data_*.txt"))):
        match = _RAW_RE.match(os.path.basename(path))
        if match:
            raw[match.group("sensor")] = path
            trial_id = match.group("trial")
    metas = glob.glob(os.path.join(trial_dir, "*_meta.json"))
    return trial_id, raw, (metas[0] if len(metas) == 1 else None)


def iter_trial_dirs(root):
    """Every trial folder below ``root`` (folders that contain a ``*_raw_data_LB.txt`` file)."""
    found = {os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", "*_raw_data_LB.txt"), recursive=True)}
    return sorted(found)


def read_sensor_file(path):
    """Read a raw sensor file. Returns a DataFrame with PacketCounter, Acc_* and Gyr_* columns."""
    with open(path, "r") as file_id:
        for skip, line in enumerate(file_id):
            if line.startswith("PacketCounter"):
                break
        else:
            raise ValueError("the column 'PacketCounter' was not found in {}".format(os.path.basename(path)))
    data = pd.read_csv(path, sep=r"[,\t; ]+", skiprows=skip, engine="python")
    data.columns = [str(c).strip() for c in data.columns]
    required = ("PacketCounter",) + ACC_COLS + GYR_COLS
    absent = [c for c in required if c not in data.columns]
    if absent:
        raise ValueError("columns {} not found in {}".format(absent, os.path.basename(path)))
    data = data[list(required)].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["PacketCounter"]).reset_index(drop=True)
    if len(data) < 10:
        raise ValueError("{} contains fewer than 10 samples".format(os.path.basename(path)))
    return data


# ---------------------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------------------

def events_table(left, right):
    """Gait events as a table (Foot, TO, HS) from lists of [toe-off, heel-strike] pairs."""
    rows = [(foot, int(to), int(hs)) for foot, pairs in ((0, left), (1, right)) for to, hs in (pairs or [])]
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["Foot", "TO", "HS"]).sort_values("HS").reset_index(drop=True)


def read_meta(path):
    """Read the metadata file of a trial of the public data sets (flat or nested layout)."""
    if path is None:
        return TrialMeta()
    with open(path, "r") as file_id:
        content = json.load(file_id)
    gait = content.get("gait") if isinstance(content.get("gait"), dict) else {}
    demo = content.get("demographics") if isinstance(content.get("demographics"), dict) else {}
    freq = content.get("freq", gait.get("freq", gait.get("sampling_frequency_hz")))
    uturn = content.get("uturnBoundaries", gait.get("uturn_boundaries_samples"))
    subject = content.get("subject", demo.get("participant_id"))
    return TrialMeta(
        freq=float(freq) if freq is not None else None,
        sensor_type=content.get("sensor", gait.get("sensor")),
        group=content.get("group"),
        pathology=content.get("pathologyKey"),
        subject=str(subject) if subject is not None else None,
        uturn_boundaries=tuple(int(v) for v in uturn) if uturn else None,
        gait_events=events_table(content.get("leftGaitEvents", gait.get("left_gait_events_samples")),
                                 content.get("rightGaitEvents", gait.get("right_gait_events_samples"))),
        content=content,
    )


# ---------------------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------------------

def _convert_units(sensors, acc_unit="auto", gyr_unit="auto"):
    """Convert to m/s^2 and rad/s. "auto": the median norm of the acceleration is close to 9.81
    in m/s^2 and to 1 in g; the angular velocity of a walking foot stays below 40 rad/s, whereas
    it commonly exceeds 300 in deg/s."""
    notes = []
    for name, data in sensors.items():
        acc = data[list(ACC_COLS)].to_numpy(dtype=float)
        in_g = acc_unit == "g" or (acc_unit == "auto" and 0.5 < np.nanmedian(np.linalg.norm(acc, axis=1)) < 2.0)
        if in_g:
            data[list(ACC_COLS)] = acc * GRAVITY
            if acc_unit == "auto":
                notes.append("{}: acceleration recognised in g and converted to m/s^2".format(name))
        gyr = data[list(GYR_COLS)].to_numpy(dtype=float)
        in_deg = gyr_unit == "deg/s" or (gyr_unit == "auto" and np.nanpercentile(np.abs(gyr), 99.5) > 40.0)
        if in_deg:
            data[list(GYR_COLS)] = np.deg2rad(gyr)
            if gyr_unit == "auto":
                notes.append("{}: angular velocity recognised in deg/s and converted to rad/s".format(name))
    return notes


# ---------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------

def load_files(files, freq, trial_id="trial", acc_unit="auto", gyr_unit="auto", meta=None, trial_dir=None):
    """Load a trial from explicit raw files.

    Arguments:
        files {dict} -- sensor ("HE", "LB", "LF", "RF") -> path; "HE" may be absent
        freq {float} -- acquisition frequency (Hz)
        acc_unit {str} -- "m/s2", "g" or "auto"
        gyr_unit {str} -- "rad/s", "deg/s" or "auto"

    Returns
    -------
    RawTrial
    """
    missing = [s for s in REQUIRED_SENSORS if not files.get(s)]
    if missing:
        raise ValueError("the raw file(s) of {} are required".format(", ".join(missing)))
    sensors = {s: read_sensor_file(files[s]) for s in SENSORS if files.get(s)}
    notes = _convert_units(sensors, acc_unit, gyr_unit)
    if "HE" not in sensors:
        notes.append("no head sensor: the head indicators are not computed")
    return RawTrial(trial_id=trial_id, freq=float(freq), sensors=sensors, meta=meta or TrialMeta(),
                    trial_dir=trial_dir, warnings=notes)


def load_trial(trial_dir, freq=None, acc_unit="auto", gyr_unit="auto"):
    """Load a trial folder of the public data sets (raw files and metadata)."""
    trial_id, raw, meta_path = find_trial_files(trial_dir)
    meta = read_meta(meta_path)
    freq = freq if freq is not None else meta.freq
    if freq is None:
        raise ValueError("the acquisition frequency is not given and not found in the metadata")
    return load_files(raw, freq, trial_id=trial_id or os.path.basename(trial_dir), acc_unit=acc_unit,
                      gyr_unit=gyr_unit, meta=meta, trial_dir=str(trial_dir))
