# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Group "raw": integrity of the data stream, per sensor, from the packet counters.

* missing       -- share of the samples lost inside the span of the sensor (%);
* max_gap       -- longest run of lost samples (s);
* disconnection -- share of the recording (span covered by the sensors together) that the
                   sensor does not cover, because it started late or stopped early (%).
"""

from ..params import SENSORS
from .base import absent, make

GROUP = "raw"


def compute(aligned, params):
    out = []
    g_first, g_last = aligned.global_span
    g_span = g_last - g_first + 1
    for s in SENSORS:
        if s not in aligned.stats:
            out += [absent(name, GROUP, s, unit) for name, unit in (("missing", "%"), ("max_gap", "s"),
                                                                      ("disconnection", "%"))]
            continue
        st = aligned.stats[s]
        missing = 100.0 * st.n_missing / st.span
        gap = st.longest_gap / aligned.freq
        disconnection = 100.0 * (g_span - st.span) / g_span
        out.append(make("missing", GROUP, s, missing, "%", missing, params, note=_missing_note(st)))
        out.append(make("max_gap", GROUP, s, gap, "s", gap, params, note=_gap_note(st, g_first, aligned.freq)))
        out.append(make("disconnection", GROUP, s, disconnection, "%", disconnection, params,
                        note=_span_note(st, g_first, g_last, aligned.freq)))
    return out


def _missing_note(st):
    if not st.n_missing:
        return ""
    return "{} samples lost in {} gap{}".format(st.n_missing, len(st.gaps), "s" if len(st.gaps) > 1 else "")


def _gap_note(st, g_first, freq):
    if not st.gaps:
        return ""
    first, length = max(st.gaps, key=lambda g: g[1])
    return "longest gap: {:.2f} s, from t = {:.1f} s".format(length / freq, (first - g_first) / freq)


def _span_note(st, g_first, g_last, freq):
    late, early = (st.first - g_first) / freq, (g_last - st.last) / freq
    parts = []
    if late > 0:
        parts.append("starts {:.2f} s after the others".format(late))
    if early > 0:
        parts.append("stops {:.2f} s before the others".format(early))
    return ", ".join(parts)
