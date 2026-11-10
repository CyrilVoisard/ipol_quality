# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import os
from datetime import datetime

from . import __version__
from .params import SENSORS
from .scoring import quality_class

#: what each segmentation reads from the raw files of the other demos
SEGMENTATION_SENSORS = {"LB": "the U-turn segmentation", "LF": "the gait events", "RF": "the gait events"}
CLASS_NAMES = {"good": "good", "acceptable": "acceptable", "critical": "critical", None: ""}


def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


def to_dict(assessment):
    """JSON-serialisable summary: scores, indicators, standing periods, segmentation, parameters."""
    a, st, seg, sc = assessment, assessment.stillness, assessment.segmentation, assessment.scores
    u = seg.uturn
    return {
        "software": "ipol_quality " + __version__,
        "date": str(datetime.now()),
        "trial": a.trial_id,
        "freq": a.aligned.freq,
        "n_samples": a.aligned.n,
        "sensors": a.aligned.sensors,
        "alignment": a.aligned.alignment,
        "warnings": a.aligned.warnings,
        "scores": {"global": _r(sc.global_score, 1),
                   "class": quality_class(sc.global_score, a.params["settings"]),
                   "groups": {g: _r(v, 1) for g, v in sc.groups.items()},
                   "group_classes": {g: quality_class(v, a.params["settings"]) for g, v in sc.groups.items()},
                   "per_sensor": {s: {g: _r(v, 1) for g, v in sc.per_sensor[s].items()} for s in SENSORS},
                   "critical": [{"group": g, "indicator": n, "sensor": s, "quality": _r(q, 1)} for g, n, s, q in sc.critical]},
        "indicators": [{"group": i.group, "name": i.name, "sensor": i.sensor, "value": _r(i.value, 4), "unit": i.unit,
                        "quality": _r(i.quality, 1), "note": i.note} for i in a.indicators],
        "standing": {"before_s": _r(st.t_before, 2), "after_s": _r(st.t_after, 2), "rest_window": st.rest.label,
                     "walk": [st.active.start, st.active.end]},
        "segmentation": {"source": seg.source,
                         "uturn": None if u is None else {"start": u.start, "end": u.end, "amplitude_deg": _r(u.amplitude, 1),
                                                          "r2": _r(u.r2, 4), "n_uturns": u.n_uturns},
                         "uturn_error": seg.uturn_error,
                         "gait_events": 0 if seg.gait_events is None else len(seg.gait_events),
                         "gait_events_error": seg.gait_events_error},
        "placement": None if not a.placement else {
            s: {"rotation_deg": p.rotation, "classic": p.classic, "mapping": p.mapping(), "description": p.describe("en")}
            for s, p in a.placement.items()},
        "placement_messages": placement_messages(a),
        "parameters": a.params,
    }


def placement_messages(assessment):
    """Sentences for the operator about the sensors turned in their plane: the axes were corrected
    before the indicators, and a segmentation computed on the raw file of such a sensor by another
    demo may be wrong."""
    a, out = assessment, []
    for s, p in (a.placement or {}).items():
        if p.classic:
            continue
        angle = {90: "+90°", 180: "180°", 270: "-90°"}[p.rotation]
        out.append("{} rotated by {} in its plane: axes corrected before the indicators "
                   "(X, Y, Z = raw {})".format(s, angle, p.mapping()))
        seg = a.segmentation
        used = (s == "LB" and seg.uturn is not None) or (s in ("LF", "RF") and seg.gait_events is not None)
        if used:
            out.append("{} feeds {}: if computed on the raw file, rerun it on corrected_{}.txt".format(
                s, SEGMENTATION_SENSORS[s], s))
    return out


def not_assessed(assessment):
    """[(name, sensors, reason)] of the indicators not assessed (left out of the scores)."""
    groups = {}
    for i in assessment.indicators:
        if i.quality is None and i.note.startswith("not assessed: "):
            key = (i.name, i.note[len("not assessed: "):])
            groups.setdefault(key, []).append(i.sensor or "")
    return [(name, ", ".join(s for s in sensors if s), reason) for (name, reason), sensors in groups.items()]


def text_report(assessment):
    """Short text report (displayed by the IPOL demo)."""
    sc = assessment.scores
    fmt = lambda v: "   -" if v is None else "{:4.0f}".format(v)
    st = assessment.params["settings"]
    cls = lambda v: "  " + CLASS_NAMES[quality_class(v, st)]    # noqa: E731
    lines = ["Recording quality score -- {}".format(assessment.trial_id), "-" * 60,
             "{:<34}{}{}   (lowest of the six groups)".format("Global score", fmt(sc.global_score), cls(sc.global_score)),
             "Classes: good above {:g}, acceptable above {:g} up to {:g}, critical up to {:g}".format(
                 st["good"], st["critical"], st["good"], st["critical"]), ""]
    names = {"raw": "Raw data", "sensor": "Sensors at rest", "orientation": "Orientation (purity)",
             "protocol": "Protocol (trial)", "uturn": "U-turn segmentation", "segmentation": "Gait events segmentation"}
    for g, v in sc.groups.items():
        lines.append("  {:<32}{}{}".format(names[g], fmt(v), cls(v)))
    lines += ["", "Per sensor            raw  sensor  orientation"]
    for s in SENSORS:
        p = sc.per_sensor[s]
        if all(v is None for v in p.values()):
            lines.append("  {:<18}  not provided".format(s))
        else:
            lines.append("  {:<18}{}    {}      {}".format(s, fmt(p["raw"]), fmt(p["sensor"]), fmt(p["orientation"])))
    if assessment.placement:
        lines += ["", "Placement of the sensors (detected, corrected before the indicators)"]
        for s, p in assessment.placement.items():
            lines.append("  {:<6}{}".format(s, p.describe("en")))
        for m in placement_messages(assessment):
            lines.append("  " + m)
    low = sorted([i for i in assessment.indicators if i.quality is not None and i.quality <= st["good"]],
                 key=lambda i: i.quality)
    if low:
        lines += ["", "Indicators at {:g} or below (critical up to {:g})".format(st["good"], st["critical"])]
        for i in low:
            lines.append("  {:<13}{:<17}{:<4}{}{}  {}".format(i.group, i.name, i.sensor or "", fmt(i.quality),
                                                            cls(i.quality), i.note))
    skipped = not_assessed(assessment)
    if skipped:
        lines += ["", "Not assessed (left out of the scores)"]
        for name, sensors, reason in skipped:
            lines.append("  {:<17}{:<16}{}".format(name, sensors, reason))
    if assessment.aligned.warnings:
        lines.append("")
        lines += ["Note: " + w for w in assessment.aligned.warnings]
    return "\n".join(lines) + "\n"


def write_reports(assessment, output, prefix=""):
    """Write quality.json, quality.txt and indicators.csv in ``output``."""
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, prefix + "quality.json"), "w") as f:
        json.dump(to_dict(assessment), f, indent=2)
    with open(os.path.join(output, prefix + "quality.txt"), "w") as f:
        f.write(text_report(assessment))
    assessment.table().to_csv(os.path.join(output, prefix + "indicators.csv"), index=False)


def save_figures(assessment, output, prefix="", dpi=110):
    """Write score.png and construction.png in ``output``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .figures import STYLE, construction_sheet, score_figure
    for name, draw in (("score", score_figure), ("construction", construction_sheet)):
        fig = draw(assessment)
        with plt.rc_context(STYLE):
            fig.savefig(os.path.join(output, prefix + name + ".png"), dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
