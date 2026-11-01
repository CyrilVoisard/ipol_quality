# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Figures of an assessment.

* :func:`score_figure` -- the score at a glance: the global score in the centre and the five
  groups around it. The upstream groups (raw data, sensors, protocol) are rings of four arcs,
  one per sensor (head top left, lower back top right, left foot bottom left, right foot bottom
  right), the U-turn and segmentation groups are progress rings. The critical indicators are
  listed with their reason.
* :func:`construction_sheet` -- the signals and intermediate results behind each indicator,
  to check visually that each one measures what it is meant to measure.

Colours: the quality goes from terracotta (below 50) through honey (62) to sage green (above
75) and its value is always written; each sensor keeps the same colour in every panel.
"""

import functools
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

from .params import FEET, GRAVITY, SENSORS
from .segmentation import lb_heading, vertical_yaw_rate

# --- palette -------------------------------------------------------------------------------
BACKGROUND = "#ffffff"           # warm alternative: "#fbf8f3"
HALO = "#f5f2ed"
INK, INK2, MUTED, TRACK, GRID = "#2f3142", "#6b6e80", "#b9b5ac", "#ece7df", "#e9e4dc"
SENSOR_COLORS = {"HE": "#3a6ea5", "LB": "#c8803a", "LF": "#2a9d8f", "RF": "#a05195"}
SENSOR_NAMES = {"HE": "head", "LB": "lower back", "LF": "left foot", "RF": "right foot"}
AXIS_COLORS = {"X": "#9fb7cc", "Y": "#e3a99a", "Z": "#b8cba0"}
_LOW, _MID, _HIGH = "#e07a5f", "#f2c14e", "#6ab187"
GROUP_TITLES = {"raw": "Raw data", "sensor": "Sensors", "protocol": "Protocol",
                "uturn": "U-turn", "segmentation": "Segmentation"}

# --- font: Inter is shipped with the code (SIL Open Font License, package/fonts/OFL.txt) so that
# the figures look the same everywhere; DejaVu Sans (matplotlib) is the fallback -----------------
_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
for _f in ("Inter-Regular.otf", "Inter-Bold.otf"):
    try:
        font_manager.fontManager.addfont(os.path.join(_FONT_DIR, _f))
    except (OSError, RuntimeError):
        pass
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Inter", "DejaVu Sans"]}


def styled(function):
    """Draw the figure with the house style (font)."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with plt.rc_context(STYLE):
            return function(*args, **kwargs)
    return wrapper


def quality_color(q):
    """Terracotta up to 50, honey at 62.5, sage green from 75; grey if not available."""
    if q is None:
        return MUTED
    if q <= 50:
        return _LOW
    if q >= 75:
        return _HIGH
    u = (q - 50) / 25.0
    a, b, v = (_LOW, _MID, 2 * u) if u < 0.5 else (_MID, _HIGH, 2 * u - 1)
    return tuple((1 - v) * np.array(mcolors.to_rgb(a)) + v * np.array(mcolors.to_rgb(b)))


def _fmt(q):
    return "–" if q is None else "{:.0f}".format(q)


# ---------------------------------------------------------------------------------------
# Score figure
# ---------------------------------------------------------------------------------------

def _arc(ax, cx, cy, r, a0, a1, color, lw, zorder=2, ls="-", alpha=1.0):
    """Arc of radius r from angle a0 to a1 (degrees, counter-clockwise) with round caps."""
    t = np.radians(np.linspace(a0, a1, max(2, int(abs(a1 - a0)))))
    ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), color=color, lw=lw, solid_capstyle="round",
            dash_capstyle="round", ls=ls, zorder=zorder, alpha=alpha)


def _progress_ring(ax, cx, cy, r, value, lw, fontsize, label=None):
    _arc(ax, cx, cy, r, 0, 360, TRACK, lw, zorder=1)
    if value is not None and value > 0.5:
        _arc(ax, cx, cy, r, 90, 90 - 3.6 * min(value, 99.7), quality_color(value), lw)
    ax.text(cx, cy, _fmt(value), ha="center", va="center", fontsize=fontsize, color=INK, weight="bold", zorder=5)
    if label:
        ax.text(cx, cy - 0.33 * r - 0.08, label, ha="center", va="top", fontsize=fontsize * 0.32, color=INK2, zorder=5)


def _sensor_ring(ax, cx, cy, r, values, center, lw, fontsize):
    """Ring of four arcs (one per sensor) with the group score in the centre."""
    centres = {"HE": 135, "LB": 45, "LF": 225, "RF": 315}
    for s, c in centres.items():
        q = values.get(s)
        if q is None:
            _arc(ax, cx, cy, r, c - 32, c + 32, MUTED, lw * 0.25, ls=(0, (1, 3)))
        else:
            _arc(ax, cx, cy, r, c - 32, c + 32, quality_color(q), lw)
        x, y = cx + (r + 0.30) * np.cos(np.radians(c)), cy + (r + 0.30) * np.sin(np.radians(c))
        ax.text(x, y, "{}\n{}".format(s, _fmt(q)), ha="center", va="center", fontsize=7.5, color=INK2,
                linespacing=1.1, zorder=5)
    ax.text(cx, cy, _fmt(center), ha="center", va="center", fontsize=fontsize, color=INK, weight="bold", zorder=5)


def _general_bar(ax, cx, y, half, value, lw):
    """Bar of the trial-level part of a group (e.g. "general" of the protocol), coloured by quality."""
    if value is None:
        ax.plot([cx - half, cx + half], [y, y], color=MUTED, lw=lw * 0.25, ls=(0, (1, 3)), dash_capstyle="round")
    else:
        ax.plot([cx - half, cx + half], [y, y], color=quality_color(value), lw=lw, solid_capstyle="round", zorder=2)
    ax.text(cx, y - 0.24, "general {}".format(_fmt(value)), ha="center", va="center", fontsize=7.5, color=INK2, zorder=5)


@styled
def score_figure(assessment, title=None):
    """The global score in the centre, the five groups around it."""
    from .scoring import combine
    sc, st = assessment.scores, assessment.params["settings"]
    crit = [i for i in assessment.indicators if i.quality is not None and i.quality < st["critical"]]
    n_max = 10
    lines = []
    for i in crit[:n_max]:
        what = "{} {}".format(i.name, i.sensor or "").strip()
        why = i.note if i.note else "value {:.3g} {}".format(i.value, i.unit) if i.value is not None else ""
        why = why if len(why) <= 64 else why[:63] + "…"
        lines.append("{:<14} {:<28} {:>3}   {}".format(GROUP_TITLES[i.group], what, _fmt(i.quality), why))
    if len(crit) > n_max:
        lines.append("… and {} more (see quality.txt)".format(len(crit) - n_max))
    from .report import not_assessed
    skipped = not_assessed(assessment)
    for name, sensors, reason in skipped:
        reason = reason if len(reason) <= 64 else reason[:63] + "…"
        what = "{} {}".format(name, sensors.replace(", ", ",")).strip()
        lines.append("{:<14} {:<28} {:>3}   {}".format("Not assessed", what[:28], "–", reason))

    # vertical layout in inches: title, star, colour scale, list of the critical indicators
    W, top, star, scale, gap, line = 8.0, 0.55, 7.35, 0.45, 0.42, 0.165
    H = top + star + scale + gap + line * max(len(lines), 1) + 0.3
    fig = plt.figure(figsize=(W, H), facecolor=BACKGROUND)
    ax = fig.add_axes([0.0, 1 - (top + star) / H, 1.0, star / H])
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-3.85, 3.9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.06, 1 - 0.3 / H, title or "Recording quality", fontsize=17, color=INK, weight="bold", va="top")

    # raw data on top, sensors left, protocol right, U-turn bottom left, segmentation bottom right
    layout = {"raw": 90, "sensor": 162, "protocol": 18, "uturn": -126, "segmentation": -54}
    R, r_sat = 2.75, 0.62
    # soft halo and spokes
    ax.add_patch(plt.Circle((0, 0), 1.45, color=HALO, zorder=0))
    for g, ang in layout.items():
        c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        ax.plot([1.42 * c, (R - r_sat - 0.42) * c], [1.42 * s, (R - r_sat - 0.42) * s],
                color=quality_color(sc.groups[g]), lw=3, alpha=0.55, solid_capstyle="round", zorder=1)
    _progress_ring(ax, 0, 0, 1.1, sc.global_score, lw=20, fontsize=44)

    for g, ang in layout.items():
        cx, cy = R * np.cos(np.radians(ang)), R * np.sin(np.radians(ang))
        ax.add_patch(plt.Circle((cx, cy), r_sat + 0.12, color=BACKGROUND, zorder=0.5))
        if g in ("raw", "sensor", "protocol"):
            _sensor_ring(ax, cx, cy, r_sat, {s: sc.per_sensor[s][g] for s in SENSORS}, sc.groups[g], lw=9, fontsize=17)
            trial = [i.quality for i in assessment.indicators if i.group == g and i.sensor is None]
            if trial:   # trial-level part of the group (protocol: number of U-turns, standing time)
                general = combine(trial, st["aggregation"], st["critical"])
                _general_bar(ax, cx, cy - r_sat - 0.62, 0.42, general, lw=9)
        else:
            _progress_ring(ax, cx, cy, r_sat, sc.groups[g], lw=9, fontsize=17)
        # titles: above the satellites of the upper half, below those of the lower half (aligned in pairs)
        dy = r_sat + 0.6 if ang > 0 else -(r_sat + 0.45)
        ax.text(cx, cy + dy, GROUP_TITLES[g], ha="center", va="center", fontsize=11.5, color=INK, weight="bold")

    # colour scale
    y_scale = 1 - (top + star + 0.12) / H
    cax = fig.add_axes([0.30, y_scale, 0.40, 0.11 / H])
    grad = np.linspace(0, 100, 256)
    cax.imshow([grad], aspect="auto", extent=(0, 100, 0, 1),
               cmap=mcolors.ListedColormap([quality_color(v) for v in grad]))
    cax.set_yticks([])
    cax.set_xticks([0, 50, 75, 100])
    cax.tick_params(labelsize=7, colors=INK2, length=0)
    for side in cax.spines.values():
        side.set_visible(False)

    # critical indicators and their reasons
    head = "Critical indicators (quality < 50)" if crit else "No critical indicator"
    if skipped:
        head += "  ·  not assessed (left out of the scores)"
    y = 1 - (top + star + scale + 0.1) / H
    fig.text(0.06, y, head, fontsize=9.5, color=INK, weight="bold", va="top")
    fig.text(0.06, y - 0.27 / H, "\n".join(lines), fontsize=7.6, color=INK2, va="top", family="monospace",
             linespacing=1.45)
    return fig


# ---------------------------------------------------------------------------------------
# Construction sheet
# ---------------------------------------------------------------------------------------

def _ind(a, name, sensor=None):
    for i in a.indicators:
        if i.name == name and i.sensor == sensor:
            return i
    return None


def _chips(ax, items, y=1.02):
    """Row of rounded labels coloured by quality above an axes: items = [(text, quality)]."""
    x = 0.0
    for text, q in items:
        w = 0.0105 * len(text) + 0.03
        ax.add_patch(FancyBboxPatch((x, y), w, 0.075, boxstyle="round,pad=0.004,rounding_size=0.03",
                                    transform=ax.transAxes, clip_on=False, facecolor=quality_color(q),
                                    edgecolor="none", alpha=0.9 if q is not None else 0.5))
        ax.text(x + w / 2, y + 0.037, text, transform=ax.transAxes, ha="center", va="center", fontsize=7,
                color=INK if q is None or q >= 50 else "white", weight="bold", clip_on=False)
        x += w + 0.012


def _panel(ax, group, title, score, chips, xlabel=None, ylabel=None):
    ax.set_facecolor("white")
    ax.text(0.0, 1.22, " {} ".format(_fmt(score)), transform=ax.transAxes, fontsize=11, color=INK, weight="bold",
            va="bottom", bbox=dict(boxstyle="round,pad=0.25", facecolor=quality_color(score), edgecolor="none", alpha=0.9))
    ax.text(0.075, 1.22, "{}  ·  {}".format(GROUP_TITLES[group], title), transform=ax.transAxes, fontsize=10.5,
            color=INK, weight="bold", va="bottom")
    _chips(ax, chips)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(labelsize=7.5, colors=INK2)
    ax.grid(True, color=GRID, linewidth=0.7)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK2)


def _sensor_chips(a, group):
    return [("{} {}".format(s, _fmt(a.scores.per_sensor[s][group])), a.scores.per_sensor[s][group]) for s in a.aligned.sensors]


def _rest_slice(a, s, window=None):
    from .indicators.sensor import rest_window
    return rest_window(a.aligned, a.stillness, s, a.params["settings"], window)


@styled
def construction_sheet(assessment):
    """Signals and intermediate results behind every indicator."""
    a, al, st, seg = assessment, assessment.aligned, assessment.stillness, assessment.segmentation
    freq, t, sensors = al.freq, al.time, al.sensors
    fig = plt.figure(figsize=(14, 16.5), facecolor=BACKGROUND)
    # reading order: raw data | sensors (2 panels) | protocol (3 panels) | U-turn | segmentation
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.15], hspace=0.62, wspace=0.16,
                          left=0.06, right=0.97, top=0.905, bottom=0.04)
    fig.text(0.06, 0.985, "Construction of the quality score", fontsize=17, color=INK, weight="bold", va="top")

    # A. data stream ------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for k, s in enumerate(SENSORS):
        y = len(SENSORS) - k
        if s not in al.stats:
            ax.text(0, y, "  not provided", va="center", fontsize=8, color=MUTED)
            continue
        stats = al.stats[s]
        ax.plot([(stats.first - al.start) / freq, (stats.last - al.start) / freq], [y, y],
                color=SENSOR_COLORS[s], lw=9, solid_capstyle="round")
        for first, length in stats.gaps:
            ax.plot([(first - al.start) / freq, (first + length - al.start) / freq], [y, y], color="black", lw=9,
                    solid_capstyle="butt", zorder=3)
        ax.annotate("{} lost".format(stats.n_missing), xy=((stats.last - al.start) / freq, y), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(SENSORS), 0, -1), [SENSOR_NAMES[s] for s in SENSORS])
    ax.set_ylim(0.4, len(SENSORS) + 0.6)
    _panel(ax, "raw", "stream of each sensor (lost samples in black)", a.scores.groups["raw"],
           _sensor_chips(a, "raw"), xlabel="time (s)")

    # B. phases ---------------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    for w, name in ((st.before, "standing"), (st.after, "standing")):
        if w is not None:
            ax.axvspan(w.start / freq, w.end / freq, color="#e3eedf", zorder=0)
            ax.text((w.start + w.end) / 2 / freq, 0.93, name, transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=7.5, color=INK2)
    ax.axvspan(st.active.start / freq, st.active.end / freq, color="#f6efe3", zorder=0)
    ax.text((st.active.start + st.active.end) / 2 / freq, 0.93, "walk", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=7.5, color=INK2)
    ax.plot(t, st.activity_acc / st.threshold_acc, color=INK, lw=1, label="acceleration")
    ax.plot(t, st.activity_gyr / st.threshold_gyr, color=INK2, lw=1, ls="--", label="angular velocity")
    ax.axhline(1, color=_LOW, lw=0.9)
    ax.set_yscale("log")
    ax.legend(fontsize=7, frameon=False, loc="lower center", ncol=2)
    i = _ind(a, "still_time")
    _panel(ax, "protocol", "activity / threshold: standing {:.1f} s before, {:.1f} s after".format(st.t_before, st.t_after),
           a.scores.groups["protocol"], [("standing time {}".format(_fmt(i.quality)), i.quality),
                                         ("U-turns {}".format(_fmt(_ind(a, "n_uturns").quality)), _ind(a, "n_uturns").quality)],
           xlabel="time (s)", ylabel="activity / threshold")

    # C. acceleration at rest ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for s in sensors:
        w = _rest_slice(a, s)
        x = np.arange(w.stop - w.start) / freq
        ax.plot(x, np.linalg.norm(al.acc[s][w], axis=1) - GRAVITY, color=SENSOR_COLORS[s], lw=1.1, label=s)
    ax.axhline(0, color=INK, lw=0.8, ls=":")
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper right")
    _panel(ax, "sensor", "|acceleration| − g on the quietest second ({} window)".format(st.rest.label),
           a.scores.groups["sensor"], [("{} {}".format(s, _fmt(_ind(a, "still_acc", s).quality)), _ind(a, "still_acc", s).quality)
                                       for s in sensors], xlabel="time in the window (s)", ylabel="m/s²")

    # D. angular velocity at rest -------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for s in sensors:
        w = _rest_slice(a, s)
        x = np.arange(w.stop - w.start) / freq
        ax.plot(x, np.degrees(np.cumsum(al.gyr[s][w], axis=0) / freq) @ np.ones(3) / np.sqrt(3), color=SENSOR_COLORS[s],
                lw=1.1, label="{}  {:.2f}°/s".format(s, _ind(a, "gyro_bias", s).value or 0))
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    _panel(ax, "sensor", "integrated angular velocity at rest (slope = bias)", a.scores.groups["sensor"],
           [("bias {} {}".format(s, _fmt(_ind(a, "gyro_bias", s).quality)), _ind(a, "gyro_bias", s).quality) for s in sensors],
           xlabel="time in the window (s)", ylabel="deg")

    # E. placement at rest ----------------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    axes_rest = a.params["settings"]["rest_axes"]
    for k, s in enumerate(SENSORS):
        if s not in al.acc:
            continue
        g = np.mean(al.acc[s][_rest_slice(a, s)], axis=0)
        share = g ** 2 / np.sum(g ** 2)
        left = 0.0
        for j, axis in enumerate("XYZ"):
            expected = axis in axes_rest[s]
            ax.barh(k, share[j], left=left, color=AXIS_COLORS[axis], height=0.55,
                    edgecolor=INK if expected else "none", linewidth=1.2 if expected else 0)
            if share[j] > 0.08:
                ax.text(left + share[j] / 2, k, "{}{}".format("−" if g[j] < 0 else "+", axis), ha="center", va="center",
                        fontsize=7.5, color=INK)
            left += share[j]
        o, stab = _ind(a, "orient_rest", s), _ind(a, "orient_stability", s)
        tilt = "Δtilt {:.1f}°".format(stab.value) if stab.value is not None else "Δtilt not assessed"
        ax.text(1.02, k, "{}   {}".format("wrong sign" if o.note.startswith("gravity") else "", tilt).strip(),
                va="center", fontsize=7.5, color=_LOW if o.note.startswith("gravity") else INK2)
    ax.set_yticks(range(len(SENSORS)), [SENSOR_NAMES[s] for s in SENSORS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 0.5, 1])
    _panel(ax, "protocol", "gravity per axis at rest (expected axes outlined)", a.scores.groups["protocol"],
           _sensor_chips(a, "protocol"), xlabel="share of the gravity")

    # F. walking plane ------------------------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    labels, k = [], 0
    act = st.active.slice()
    for s in sensors:
        g = np.mean(al.acc[s][_rest_slice(a, s)], axis=0)
        e = np.mean((al.acc[s][act] - g) ** 2, axis=0)
        rows = [("acceleration " + s, e / np.sum(e), ("X", "Z"))]
        if s in FEET:
            eg = np.mean(al.gyr[s][act] ** 2, axis=0)
            rows.append(("rotation " + s, eg / np.sum(eg), ("Y",)))
        for name, share, good in rows:
            left = 0.0
            for j, axis in enumerate("XYZ"):
                ax.barh(k, share[j], left=left, color=AXIS_COLORS[axis], height=0.55,
                        edgecolor=INK if axis in good else "none", linewidth=1.2 if axis in good else 0)
                if share[j] > 0.08:
                    ax.text(left + share[j] / 2, k, axis, ha="center", va="center", fontsize=7.5, color=INK)
                left += share[j]
            labels.append(name)
            k += 1
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    chips = [("acc {} {}".format(s, _fmt(_ind(a, "walk_plane_acc", s).quality)), _ind(a, "walk_plane_acc", s).quality) for s in sensors]
    chips += [("rot {} {}".format(s, _fmt(_ind(a, "walk_plane_gyr", s).quality)), _ind(a, "walk_plane_gyr", s).quality) for s in FEET]
    _panel(ax, "protocol", "energy of the walk per axis (expected axes outlined)", a.scores.groups["protocol"], chips,
           xlabel="share of the energy")

    # G. U-turn -------------------------------------------------------------------------------------------
    ax = fig.add_subplot(gs[3, 0])
    u = seg.uturn
    chips = [(lab + " " + _fmt(_ind(a, n).quality), _ind(a, n).quality)
             for lab, n in (("fit", "uturn_fit"), ("amplitude", "uturn_amplitude"), ("residual drift", "straight_dev"),
                            ("head/trunk", "head_trunk"))]
    if u is not None:
        theta, model = lb_heading(al, u)
        ax.axvspan(u.start / freq, u.end / freq, color="#f6efe3", zorder=0)
        ax.plot(t, theta, color=SENSOR_COLORS["LB"], lw=1.4, label="lower back (model input)")
        if "HE" in al.acc:
            yaw_he = vertical_yaw_rate(al, "HE", _rest_slice(a, "HE"))
            yaw_lb = vertical_yaw_rate(al, "LB", _rest_slice(a, "LB"))
            sign = np.sign(np.sum(yaw_lb[u.start:u.end]) * (theta[min(u.end, al.n - 1)] - theta[max(u.start, 0)])) or 1
            th = sign * np.cumsum(yaw_he) / freq
            th = th - np.median(th[max(u.start - int(freq), 0):max(u.start, 1)]) + np.median(theta[max(u.start - int(freq), 0):max(u.start, 1)])
            ax.plot(t, th, color=SENSOR_COLORS["HE"], lw=1.1, label="head (about the vertical)")
        ax.plot(np.arange(*u.fit_window[:2]) / freq if len(model) == u.fit_window[1] - u.fit_window[0] else
                np.arange(len(model)) / freq + max(u.fit_window[0], 0) / freq, model, color=INK, lw=1.3, ls="--", label="fitted model")
        for b in (u.start, u.end):
            ax.axvline(b / freq, color=INK, lw=0.8)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    else:
        ax.text(0.5, 0.5, "U-turn not segmented\n" + str(seg.uturn_error), transform=ax.transAxes, ha="center",
                va="center", fontsize=9, color=_LOW)
    _panel(ax, "uturn", "", a.scores.groups["uturn"], chips, xlabel="time (s)", ylabel="deg")

    # H. gait events ---------------------------------------------------------------------------------------
    ax = fig.add_subplot(gs[3, 1])
    ev = seg.gait_events
    offset = {0: 0.0, 1: -1.0}
    for foot, s in ((0, "LF"), (1, "RF")):
        gy = np.degrees(al.gyr[s][:, 1])
        scale = np.percentile(np.abs(gy), 99) or 1.0
        ax.plot(t, gy / scale / 2.2 + offset[foot], color=SENSOR_COLORS[s], lw=0.9)
        ax.text(-0.01, offset[foot], s, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8,
                color=SENSOR_COLORS[s], weight="bold")
        if ev is not None:
            for r in ev[ev.Foot == foot].itertuples():
                ax.axvspan(r.TO / freq, r.HS / freq, ymin=(offset[foot] + 1.5) / 2.0 - 0.24,
                           ymax=(offset[foot] + 1.5) / 2.0 + 0.24, color=SENSOR_COLORS[s], alpha=0.16, lw=0)
    if u is not None:
        ax.axvspan(u.start / freq, u.end / freq, color="#f6efe3", zorder=0)
    ax.set_ylim(-1.5, 0.5)
    ax.set_yticks([])
    i = _ind(a, "alternation")
    n_ev = 0 if ev is None else len(ev)
    _panel(ax, "segmentation", "{} events".format(n_ev),
           a.scores.groups["segmentation"], [("alternation " + _fmt(i.quality), i.quality)], xlabel="time (s)")
    if ev is None:
        ax.text(0.5, 0.5, "no gait events\n" + str(seg.gait_events_error), transform=ax.transAxes, ha="center",
                va="center", fontsize=9, color=_LOW)
    return fig
