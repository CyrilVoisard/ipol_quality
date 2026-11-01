# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import dataclass

import pandas as pd

from . import indicators as ind
from .io import load_files, load_trial
from .params import default_params
from .scoring import aggregate
from .segmentation import Segmentation, from_metadata, read_segmentation
from .stillness import detect_stillness
from .timebase import align


@dataclass
class Assessment:
    trial_id: str
    aligned: object
    stillness: object
    segmentation: object
    indicators: list
    scores: object
    params: dict

    def table(self):
        """One row per indicator (and per sensor)."""
        return pd.DataFrame([vars(i) for i in self.indicators])

    def summary(self):
        """Group and global scores as a flat dictionary."""
        out = {"trial": self.trial_id, "global": self.scores.global_score}
        out.update(self.scores.groups)
        for s, groups in self.scores.per_sensor.items():
            for g, v in groups.items():
                out["{}_{}".format(s, g)] = v
        return out


def assess(raw_trial, segmentation=None, params=None):
    """Score a loaded trial.

    Arguments:
        raw_trial {RawTrial} -- output of :func:`gait_sqi.io.load_trial` or ``load_files``
        segmentation -- segmentation file (path or dictionary), "metadata" (gait events of the
            public data sets, no U-turn model), a :class:`Segmentation`, or None (no segmentation:
            the U-turn and segmentation groups are 0)
        params {dict} -- see :func:`params.default_params`

    Returns
    -------
    Assessment
    """
    params = params or default_params()
    aligned = align(raw_trial)
    if isinstance(segmentation, Segmentation):
        seg = segmentation
    elif segmentation == "metadata":
        seg = from_metadata(aligned)
    elif segmentation is None:
        seg = Segmentation(uturn_error="no segmentation given", gait_events_error="no segmentation given")
    else:
        try:
            seg = read_segmentation(segmentation, aligned)
        except (OSError, ValueError) as error:     # unreadable file: scored, with the reason
            reason = "unreadable segmentation file ({})".format(error)
            seg = Segmentation(uturn_error=reason, gait_events_error=reason)
    stillness = detect_stillness(aligned, params["settings"])
    indicators = ind.compute_all(aligned, stillness, seg, params)
    return Assessment(trial_id=raw_trial.trial_id, aligned=aligned, stillness=stillness, segmentation=seg,
                      indicators=indicators, scores=aggregate(indicators, params), params=params)


def assess_trial(trial_dir, segmentation=None, params=None, freq=None, acc_unit="auto", gyr_unit="auto"):
    """Score a trial folder of the public data sets (see :func:`assess`)."""
    return assess(load_trial(trial_dir, freq=freq, acc_unit=acc_unit, gyr_unit=gyr_unit),
                  segmentation=segmentation, params=params)


def assess_files(files, segmentation, freq, trial_id="trial", params=None, acc_unit="auto", gyr_unit="auto"):
    """Score a trial given as explicit raw files (``files``: sensor -> path, "HE" optional)."""
    return assess(load_files(files, freq, trial_id=trial_id, acc_unit=acc_unit, gyr_unit=gyr_unit),
                  segmentation=segmentation, params=params)
