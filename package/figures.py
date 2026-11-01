# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later

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
GROUP_TITLES = {"raw": "Raw data", "sensor": "Sensors", "orientation": "Orientation", "protocol": "Protocol",
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
    ax.text(cx, cy, _fmt(value), ha="center", va="center", fontsize=fontsize, weight="bold", zorder=5,
            color=_LOW if value is not None and value < 50 else INK)
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


def _critical_lines(assessment, n_max=10):
    """Text lines of the critical and not assessed indicators, and the heading of the list."""
    st = assessment.params["settings"]
    crit = [i for i in assessment.indicators if i.quality is not None and i.quality < st["critical"]]
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
    head = "Critical indicators (quality < 50)" if crit else "No critical indicator"
    if skipped:
        head += "  ·  not assessed (left out of the scores)"
    return head, lines


def _scale_and_list(fig, assessment, y_scale_in, H, x0=0.06, scale_x=(0.30, 0.40)):
    """Colour scale and list of the critical indicators under a score figure (y in inches from top)."""
    cax = fig.add_axes([scale_x[0], 1 - (y_scale_in + 0.12) / H, scale_x[1], 0.11 / H])
    grad = np.linspace(0, 100, 256)
    cax.imshow([grad], aspect="auto", extent=(0, 100, 0, 1),
               cmap=mcolors.ListedColormap([quality_color(v) for v in grad]))
    cax.set_yticks([])
    cax.set_xticks([0, 50, 75, 100])
    cax.tick_params(labelsize=7, colors=INK2, length=0)
    for side in cax.spines.values():
        side.set_visible(False)
    head, lines = _critical_lines(assessment)
    y = 1 - (y_scale_in + 0.55) / H
    fig.text(x0, y, head, fontsize=9.5, color=INK, weight="bold", va="top")
    fig.text(x0, y - 0.27 / H, "\n".join(lines), fontsize=7.6, color=INK2, va="top", family="monospace",
             linespacing=1.45)


@styled
def score_figure(assessment, title=None):
    """The global score in the centre, the six groups around it (star layout)."""
    sc, st = assessment.scores, assessment.params["settings"]
    head, lines = _critical_lines(assessment)

    # vertical layout in inches: title, star, colour scale, list of the critical indicators
    W, top, star, scale, gap, line = 8.0, 0.55, 7.6, 0.45, 0.42, 0.165
    H = top + star + scale + gap + line * max(len(lines), 1) + 0.3
    fig = plt.figure(figsize=(W, H), facecolor=BACKGROUND)
    ax = fig.add_axes([0.0, 1 - (top + star) / H, 1.0, star / H])
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-4.15, 3.9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.06, 1 - 0.3 / H, title or "Recording quality", fontsize=17, color=INK, weight="bold", va="top")

    # raw data on top, sensors and placement on both sides, U-turn and segmentation below, protocol
    # (trial level) at the bottom
    layout = {"raw": 90, "sensor": 150, "orientation": 30, "uturn": -150, "segmentation": -30, "protocol": -90}
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
        if g in ("raw", "sensor", "orientation"):
            _sensor_ring(ax, cx, cy, r_sat, {s: sc.per_sensor[s][g] for s in SENSORS}, sc.groups[g], lw=9, fontsize=17)
        else:
            _progress_ring(ax, cx, cy, r_sat, sc.groups[g], lw=9, fontsize=17)
        # titles: above the satellites of the upper half, below those of the lower half (aligned in pairs)
        dy = r_sat + 0.6 if ang > 0 else -(r_sat + 0.45)
        if g == "protocol":
            dy = -(r_sat + 0.4)
        ax.text(cx, cy + dy, GROUP_TITLES[g], ha="center", va="center", fontsize=11.5, color=INK, weight="bold")

    _scale_and_list(fig, assessment, top + star, H)
    return fig


# ---------------------------------------------------------------------------------------
# Score figure, body layout: the sensors on a schematic body, the trial on the right
# ---------------------------------------------------------------------------------------

SENSOR_TITLES = {"HE": "Head", "LB": "Lower back", "LF": "Left foot", "RF": "Right foot"}
SKIN = "#ebe8e3"


def _pill(ax, x, y, w, h, text, q, fontsize=7.6):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size={}".format(h / 2),
                                facecolor=quality_color(q), edgecolor="none", alpha=0.95 if q is not None else 0.5,
                                zorder=4))
    ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=fontsize, weight="bold", zorder=5,
            color=INK if q is None or q >= 50 else "white")


def _glyph(ax, x, y, theta, size=0.34):
    """Axes of a sensor in the picture plane: X (up for a classic sensor), Y, and Z towards the
    viewer; theta = rotation of the sensor in its plane (degrees)."""
    for name, base, color in (("X", 90.0, INK), ("Y", 180.0, INK2)):
        a = np.radians(base + theta)
        dx, dy = size * np.cos(a), size * np.sin(a)
        ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, shrinkA=0, shrinkB=0, mutation_scale=8))
        ax.text(x + 1.32 * dx, y + 1.32 * dy, name, ha="center", va="center", fontsize=7, color=color, zorder=6,
                weight="bold")
    ax.plot(x, y, "o", ms=7, mfc="white", mec=INK, mew=1.1, zorder=6)
    ax.plot(x, y, "o", ms=2, color=INK, zorder=7)
    ax.text(x + 0.17, y - 0.2, "Z", fontsize=6.5, color=INK2, zorder=6)


def _body(ax, cx, y0, k):
    """Schematic body seen from the back: centre cx, feet at y0, scale k (figure units)."""
    P = lambda bx, by: (cx + k * bx, y0 + k * by)        # noqa: E731
    patch = lambda xy, w, h, r: ax.add_patch(FancyBboxPatch(  # noqa: E731
        P(*xy), k * w, k * h, boxstyle="round,pad=0,rounding_size={}".format(k * r), facecolor=SKIN,
        edgecolor="none", zorder=1))
    patch((-1.55, 5.2), 3.1, 4.3, 0.7)            # trunk
    patch((-2.25, 5.6), 0.55, 3.7, 0.27)          # arms
    patch((1.7, 5.6), 0.55, 3.7, 0.27)
    patch((-1.25, 0.55), 1.0, 5.0, 0.45)          # legs
    patch((0.25, 0.55), 1.0, 5.0, 0.45)
    patch((-0.35, 9.4), 0.7, 0.6, 0.2)            # neck
    ax.add_patch(plt.Circle(P(0, 10.65), k * 0.95, color=SKIN, zorder=1))
    ax.add_patch(FancyBboxPatch(P(-1.45, 0.0), k * 1.4, k * 0.6, boxstyle="round,pad=0,rounding_size={}".format(k * 0.3),
                                facecolor=SKIN, edgecolor="none", zorder=1))
    ax.add_patch(FancyBboxPatch(P(0.05, 0.0), k * 1.4, k * 0.6, boxstyle="round,pad=0,rounding_size={}".format(k * 0.3),
                                facecolor=SKIN, edgecolor="none", zorder=1))
    return {"HE": P(0, 10.65), "LB": P(0, 5.9), "LF": P(-0.75, 0.3), "RF": P(0.75, 0.3)}


def _sensor_card(ax, x, y, s, assessment, w=2.25, h=1.02):
    sc = assessment.scores.per_sensor[s]
    present = s in assessment.aligned.acc
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="white", edgecolor="#dedad3", lw=1.0, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y - h / 2), 0.07, h, boxstyle="square,pad=0", facecolor=SENSOR_COLORS[s],
                                edgecolor="none", zorder=4))
    ax.text(x + 0.18, y + h / 2 - 0.2, "{} ({})".format(SENSOR_TITLES[s], s), fontsize=9, weight="bold",
            color=SENSOR_COLORS[s], va="center", zorder=5)
    if not present:
        ax.text(x + 0.18, y - 0.12, "not provided", fontsize=8, color=MUTED, va="center", zorder=5)
        return
    p = (assessment.placement or {}).get(s)
    if p is None:
        status, color = "frame assumed classic", INK2
    elif p.classic:
        status, color = "classic placement", "#3f8a5f"
    else:
        status = "rotated {} in its plane, corrected".format({90: "+90°", 180: "180°", 270: "−90°"}[p.rotation])
        color = "#a7752a"
    ax.text(x + 0.18, y + h / 2 - 0.45, status, fontsize=7.4, color=color, va="center", zorder=5)
    pw, gap = 0.62, 0.07
    for k, (g, label) in enumerate((("raw", "raw"), ("sensor", "rest"), ("orientation", "orient."))):
        _pill(ax, x + 0.18 + k * (pw + gap), y - h / 2 + 0.24, pw, 0.27, "{} {}".format(label, _fmt(sc[g])), sc[g],
              fontsize=6.6)


@styled
def sensor_figure(assessment, title=None):
    """Score at a glance, body layout. Left: the four sensors on a schematic body seen from the back
    (placement detected and corrected, scores of the raw data, of the rest and of the orientation of
    each sensor). Centre: the three sensor-level groups, the global score (mean of the six groups) and
    the three trial-level groups (protocol, U-turn, segmentation)."""
    sc = assessment.scores
    head, lines = _critical_lines(assessment)
    W, top, main, scale, gap, line = 15.5, 0.55, 7.2, 0.45, 0.42, 0.165
    H = top + main + scale + gap + line * max(len(lines), 1) + 0.3
    fig = plt.figure(figsize=(W, H), facecolor=BACKGROUND)
    ax = fig.add_axes([0.0, 1 - (top + main) / H, 1.0, main / H])
    ax.set_xlim(0, W)
    ax.set_ylim(0, main)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.035, 1 - 0.3 / H, title or "Recording quality", fontsize=17, color=INK, weight="bold", va="top")

    # --- sensors on the body ------------------------------------------------------------------------
    cx = 3.55
    pos = _body(ax, cx, 0.35, 0.56)
    ax.text(cx, main - 0.25, "Sensors", ha="center", fontsize=12, weight="bold", color=INK)
    cards = {"HE": (cx + 1.25, 6.05), "LB": (cx - 3.45, 3.95), "LF": (cx - 3.45, 1.15), "RF": (cx + 1.25, 1.15)}
    for s in SENSORS:
        x, y = pos[s]
        card_x, card_y = cards[s]
        anchor = (card_x, card_y) if card_x > cx else (card_x + 2.25, card_y)
        ax.plot([x, anchor[0]], [y, anchor[1]], color=SENSOR_COLORS[s], lw=1.0, alpha=0.6, zorder=2)
        if s in assessment.aligned.acc:
            p = (assessment.placement or {}).get(s)
            ax.add_patch(plt.Circle((x, y), 0.36, color=SENSOR_COLORS[s], alpha=0.18, zorder=2))
            _glyph(ax, x, y, p.rotation if p is not None else 0)
        else:
            ax.add_patch(plt.Circle((x, y), 0.25, facecolor="white", edgecolor=MUTED, ls=(0, (1, 2)), zorder=2))
        _sensor_card(ax, card_x, card_y, s, assessment)

    # --- the six groups around the global score ----------------------------------------------------------
    gx, gy, lx, tx = 10.9, 3.55, 8.35, 13.3
    ys = (5.75, 3.55, 1.35)
    ax.add_patch(plt.Circle((gx, gy), 1.38, color=HALO, zorder=0))
    _progress_ring(ax, gx, gy, 1.05, sc.global_score, lw=19, fontsize=42)
    ax.text(gx, gy - 1.62, "mean of the six groups", ha="center", va="center", fontsize=8, color=INK2)
    ax.text(lx, main - 0.25, "All sensors", ha="center", fontsize=12, weight="bold", color=INK)
    ax.text(tx, main - 0.25, "Trial", ha="center", fontsize=12, weight="bold", color=INK)
    for column, x0, side in (((("raw", "Raw data"), ("sensor", "At rest"), ("orientation", "Orientation")), lx, -1),
                             ((("protocol", "Protocol"), ("uturn", "U-turn"), ("segmentation", "Segmentation")), tx, 1)):
        for (g, label), y in zip(column, ys):
            ax.plot([gx + side * 1.42, x0 - side * 0.62], [gy, y], color=quality_color(sc.groups[g]), lw=2.5,
                    alpha=0.5, solid_capstyle="round", zorder=0)
            ax.add_patch(plt.Circle((x0, y), 0.66, color=BACKGROUND, zorder=0.5))
            _progress_ring(ax, x0, y, 0.55, sc.groups[g], lw=8, fontsize=15)
            if side > 0:
                ax.text(x0 + 0.8, y, label, ha="left", va="center", fontsize=10.5, weight="bold", color=INK)
            else:
                ax.text(x0, y - 0.86, label, ha="center", va="center", fontsize=10.5, weight="bold", color=INK)

    _scale_and_list(fig, assessment, top + main, H, x0=0.035, scale_x=(0.36, 0.28))
    return fig


# ---------------------------------------------------------------------------------------
# Construction sheet
# ---------------------------------------------------------------------------------------

def _ind(a, name, sensor=None):
    for i in a.indicators:
        if i.name == name and i.sensor == sensor:
            return i
    return None


def _chips(ax, items):
    """Row of rounded labels coloured by quality just above an axes: items = [(text, quality)]."""
    x = 0.0
    for text, q in items:
        w = 4.35 * len(text) + 9.0                      # points
        ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(x + w / 2, 9), textcoords="offset points",
                    ha="center", va="center", fontsize=7, weight="bold", annotation_clip=False,
                    color=INK if q is None or q >= 50 else "white",
                    bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.75", facecolor=quality_color(q),
                              edgecolor="none", alpha=0.9 if q is not None else 0.5))
        x += w + 5.0


def _panel(ax, group, title, score, chips, xlabel=None, ylabel=None):
    ax.set_facecolor("white")
    ax.annotate(" {} ".format(_fmt(score)), xy=(0, 1), xycoords="axes fraction", xytext=(0, 26),
                textcoords="offset points", fontsize=11, color=INK, weight="bold", va="bottom",
                annotation_clip=False, bbox=dict(boxstyle="round,pad=0.25", facecolor=quality_color(score),
                                                 edgecolor="none", alpha=0.9))
    ax.annotate("{}  ·  {}".format(GROUP_TITLES[group], title) if title else GROUP_TITLES[group], xy=(0, 1),
                xycoords="axes fraction", xytext=(34, 26), textcoords="offset points", fontsize=10.5, color=INK,
                weight="bold", va="bottom", annotation_clip=False)
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
    fig = plt.figure(figsize=(14, 17), facecolor=BACKGROUND)
    # as the score figure: the sensors on the left (raw data, rest, orientation), the trial on the
    # right (protocol, U-turn, segmentation)
    outer = fig.add_gridspec(1, 2, wspace=0.17, left=0.07, right=0.97, top=0.905, bottom=0.035)
    col_s = outer[0, 0].subgridspec(5, 1, hspace=0.5)
    col_t = outer[0, 1].subgridspec(3, 1, height_ratios=[0.8, 1.1, 1.1], hspace=0.25)
    fig.text(0.035, 0.985, "Construction of the quality score", fontsize=17, color=INK, weight="bold", va="top")
    for x, head in ((0.07, "Sensors"), (0.555, "Trial")):
        fig.text(x, 0.945, head, fontsize=13, color=INK, weight="bold", va="center")

    # A. data stream ------------------------------------------------------------------------
    ax = fig.add_subplot(col_s[0])
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
    ax = fig.add_subplot(col_t[0])
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
    _panel(ax, "protocol", "standing {:.1f} s before the walk, {:.1f} s after".format(st.t_before, st.t_after),
           a.scores.groups["protocol"], [("standing time {}".format(_fmt(i.quality)), i.quality),
                                         ("U-turns {}".format(_fmt(_ind(a, "n_uturns").quality)), _ind(a, "n_uturns").quality)],
           xlabel="time (s)", ylabel="activity / threshold")

    # C. acceleration at rest ------------------------------------------------------------------
    ax = fig.add_subplot(col_s[1])
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
    ax = fig.add_subplot(col_s[2])
    for s in sensors:
        w = _rest_slice(a, s)
        x = np.arange(w.stop - w.start) / freq
        ax.plot(x, np.degrees(np.cumsum(al.gyr[s][w], axis=0) / freq) @ np.ones(3) / np.sqrt(3), color=SENSOR_COLORS[s],
                lw=1.1, label="{}  {:.2f}°/s".format(s, _ind(a, "gyro_bias", s).value or 0))
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    _panel(ax, "sensor", "integrated angular velocity at rest (slope = bias)", a.scores.groups["sensor"],
           [("bias {} {}".format(s, _fmt(_ind(a, "gyro_bias", s).quality)), _ind(a, "gyro_bias", s).quality) for s in sensors],
           xlabel="time in the window (s)", ylabel="deg")

    # E. orientation at rest, placement detected ------------------------------------------------------
    ax = fig.add_subplot(col_s[3])
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
        p = (a.placement or {}).get(s)
        if o.note.startswith("gravity along -"):
            where, color = "wrong sign", _LOW
        elif p is None:
            where, color = "frame assumed classic", INK2
        elif p.classic:
            where, color = "classic placement", "#3f8a5f"
        else:
            where, color = "rotated {}, corrected".format({90: "+90°", 180: "180°", 270: "−90°"}[p.rotation]), "#a7752a"
        tilt = "Δtilt {:.1f}°".format(stab.value) if stab.value is not None else "Δtilt –"
        ax.text(1.03, k, where, va="center", fontsize=7.5, color=color)
        ax.text(1.03, k + 0.32, tilt, va="center", fontsize=6.8, color=INK2)
    ax.set_yticks(range(len(SENSORS)), [SENSOR_NAMES[s] for s in SENSORS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.42)
    ax.set_xticks([0, 0.5, 1])
    _panel(ax, "orientation", "gravity per axis at rest (expected axes outlined)", a.scores.groups["orientation"],
           _sensor_chips(a, "orientation"), xlabel="share of the gravity")

    # F. walking plane ------------------------------------------------------------------------------------
    ax = fig.add_subplot(col_s[4])
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
    _panel(ax, "orientation", "energy of the walk per axis (expected axes outlined)", a.scores.groups["orientation"], chips,
           xlabel="share of the energy")

    # G. U-turn -------------------------------------------------------------------------------------------
    ax = fig.add_subplot(col_t[1])
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
    ax = fig.add_subplot(col_t[2])
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
