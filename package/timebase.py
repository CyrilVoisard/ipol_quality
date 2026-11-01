# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Integrity of the data stream and common time base of the four sensors.

Each raw file carries a packet counter that is incremented once per sample. A jump of k > 1
between two consecutive rows reveals k - 1 lost samples. The XSens counter is coded on 16
bits and wraps around after 65535: it is unwrapped first.

When the counters of the four sensors cover a common range (synchronised sensors, as with
the XSens MTw Awinda), the sensors are aligned on their counter values. Otherwise each
counter is expressed from its first value ("first_sample" alignment). The four signals are
then linearly interpolated on the common integer grid (intersection of the four spans), so
that sample i corresponds to time i / F for every sensor.
"""

from dataclasses import dataclass, field

import numpy as np

from .io import ACC_COLS, GYR_COLS

COUNTER_MODULUS = 65536


@dataclass
class StreamStats:
    """Integrity of the stream of one sensor, from its unwrapped packet counter."""
    first: int            # first counter value (in the aligned counter scale)
    last: int             # last counter value
    n_received: int       # number of rows of the file
    n_missing: int        # number of samples lost inside [first, last]
    longest_gap: int      # longest run of lost samples
    n_disordered: int     # rows whose counter does not increase (duplicates, resets)
    gaps: list = field(default_factory=list)   # (first lost counter value, number of lost samples)

    @property
    def span(self):
        """Number of samples between the first and the last counter, included."""
        return self.last - self.first + 1


@dataclass
class AlignedTrial:
    """Raw signals of the four sensors on a common time base (SI units, not filtered)."""
    trial_id: str
    freq: float
    acc: dict                 # sensor -> (n, 3) array, m/s^2 (sensors present only)
    gyr: dict                 # sensor -> (n, 3) array, rad/s
    start: int                # counter value of sample 0
    alignment: str            # "counter" or "first_sample"
    stats: dict               # sensor -> StreamStats
    global_span: tuple        # (first, last) counter values over the four sensors
    meta: object = None
    warnings: list = field(default_factory=list)

    @property
    def sensors(self):
        """Sensors present in the trial, in the order HE, LB, LF, RF."""
        return [s for s in ("HE", "LB", "LF", "RF") if s in self.acc]

    @property
    def n(self):
        return len(self.acc["LB"])

    def to_grid(self, sensor, index):
        """Sample index of the common grid of an index of the segmentation files: samples are
        counted from the first packet counter of the file of ``sensor``, lost packets included
        (both IPOL segmentation demos resample the file on its packet counter)."""
        return int(round(self.stats[sensor].first + index - self.start))

    def to_file(self, sensor, index):
        """Inverse of :meth:`to_grid`."""
        return int(round(index + self.start - self.stats[sensor].first))

    @property
    def time(self):
        return np.arange(self.n) / self.freq


def unwrap_counter(counter, modulus=COUNTER_MODULUS):
    """Unwrap a counter coded modulo ``modulus`` (a drop larger than half the modulus is a wrap)."""
    counter = np.asarray(counter, dtype=float)
    wraps = np.diff(counter) < -modulus / 2
    return counter + modulus * np.concatenate(([0.0], np.cumsum(wraps)))


def stream_stats(counter):
    """Lost samples and disordered rows of an unwrapped counter."""
    counter = np.round(np.asarray(counter, dtype=float))   # counters written as 7.000000000000001
    steps = np.diff(counter)
    where = np.where(steps > 1)[0]
    gaps = steps[where] - 1
    return StreamStats(first=int(counter[0]), last=int(np.max(counter)), n_received=len(counter),
                       n_missing=int(np.sum(gaps)), longest_gap=int(np.max(gaps)) if len(gaps) else 0,
                       n_disordered=int(np.sum(steps <= 0)),
                       gaps=[(int(counter[i]) + 1, int(g)) for i, g in zip(where, gaps)])


def align(raw_trial, min_overlap=0.5):
    """Put the raw signals of the sensors of a trial on a common time base.

    Arguments:
        raw_trial {RawTrial} -- output of :func:`gait_sqi.io.load_trial`
        min_overlap {float} -- minimal overlap of the counter spans, as a fraction of the
            shortest span, for the counters to be considered as synchronised

    Returns
    -------
    AlignedTrial
    """
    names = list(raw_trial.sensors)
    counters = {s: unwrap_counter(raw_trial.sensors[s]["PacketCounter"]) for s in names}

    # a wrap that occurred before the first row of some files only
    firsts = np.array([c[0] for c in counters.values()])
    if firsts.max() - firsts.min() > COUNTER_MODULUS / 2:
        for s, c in counters.items():
            if c[0] < firsts.max() - COUNTER_MODULUS / 2:
                counters[s] = c + COUNTER_MODULUS

    spans = {s: (c[0], np.max(c)) for s, c in counters.items()}
    overlap = min(e for _, e in spans.values()) - max(b for b, _ in spans.values())
    shortest = min(e - b for b, e in spans.values())
    alignment = "counter" if overlap >= min_overlap * shortest else "first_sample"
    warnings = list(raw_trial.warnings)
    if alignment == "first_sample":
        counters = {s: c - c[0] for s, c in counters.items()}
        warnings.append("packet counters not synchronised: sensors aligned on their first sample")

    stats = {s: stream_stats(c) for s, c in counters.items()}
    start = max(st.first for st in stats.values())
    end = min(st.last for st in stats.values())
    if end - start + 1 < 2 * raw_trial.freq:
        raise ValueError("the sensors overlap on less than 2 s")
    grid = np.arange(start, end + 1, dtype=float)

    acc, gyr = {}, {}
    for s in names:
        # strictly increasing counter: the first row of each counter value is kept
        values, index = np.unique(counters[s], return_index=True)
        data = raw_trial.sensors[s].iloc[index]
        acc[s] = np.column_stack([np.interp(grid, values, data[c].to_numpy(float)) for c in ACC_COLS])
        gyr[s] = np.column_stack([np.interp(grid, values, data[c].to_numpy(float)) for c in GYR_COLS])

    global_span = (min(st.first for st in stats.values()), max(st.last for st in stats.values()))
    return AlignedTrial(trial_id=raw_trial.trial_id, freq=raw_trial.freq, acc=acc, gyr=gyr,
                        start=int(start), alignment=alignment, stats=stats, global_span=global_span,
                        meta=raw_trial.meta, warnings=warnings)
