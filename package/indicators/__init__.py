# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The indicators of the five groups of the score."""

from . import protocol, raw, sensor, steps, uturn
from .base import GROUPS, Indicator


def compute_all(aligned, stillness, segmentation, params):
    """Every indicator of a trial, in the order of the groups."""
    return (raw.compute(aligned, params)
            + sensor.compute(aligned, stillness, segmentation, params)
            + protocol.compute(aligned, stillness, segmentation, params)
            + uturn.compute(aligned, stillness, segmentation, params)
            + steps.compute(segmentation, params))


__all__ = ["GROUPS", "Indicator", "compute_all"]
