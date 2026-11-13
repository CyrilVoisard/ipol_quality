#!/usr/bin/env python3
# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later
"""IPOL demo: recording quality score of a 10 m walk / U-turn / 10 m walk trial.

Inputs (raw files of the sensors, columns PacketCounter, Acc_X/Y/Z, Gyr_X/Y/Z):
    -i0  head        (optional: the file may be absent)
    -i1  lower back
    -i2  left foot
    -i3  right foot
    -i4  segmentation file: {"uturn": <uturn.json of the U-turn demo>,
                             "gait_events": <gait_events.json of the MP-DTW demo>}

    python main.py -i0 input_0.txt -i1 input_1.txt -i2 input_2.txt -i3 input_3.txt \\
                   -i4 input_4.json -freq 100 -acc_unit m/s2 -gyr_unit rad/s

Outputs (in the current folder):
    score.png          the score at a glance
    construction.png   the signals behind every indicator
    quality.txt        scores, placement and indicators at 75 or below (displayed by the demo)
    quality.json       every value, indicator and parameter (download)
    indicators.csv     one row per indicator (download)
    corrected_XX.txt   raw file of each sensor with its axes turned into the classic frame and, for the
                       lower back, X aligned on the vertical (download): input of the U-turn and
                       gait-event demos and of the gait parameters
    algo_info.txt      head=0/1 (used by the demo page)
    demo_failure.txt   reason of the failure, if the trial cannot be scored
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, ROOT)

from package.params import default_params  # noqa: E402
from package.pipeline import assess_files  # noqa: E402
from package.placement import write_corrected  # noqa: E402
from package.report import save_figures, write_reports  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Recording quality score of a 10 m walk / U-turn trial.")
    parser.add_argument("-i0", help="raw file of the head sensor (optional)")
    parser.add_argument("-i1", required=True, help="raw file of the lower-back sensor")
    parser.add_argument("-i2", required=True, help="raw file of the left foot sensor")
    parser.add_argument("-i3", required=True, help="raw file of the right foot sensor")
    parser.add_argument("-i4", help="segmentation file (JSON)")
    parser.add_argument("-freq", type=float, default=100.0, help="acquisition frequency (Hz)")
    parser.add_argument("-acc_unit", default="m/s2", choices=["m/s2", "g", "auto"])
    parser.add_argument("-gyr_unit", default="rad/s", choices=["rad/s", "deg/s", "auto"])
    parser.add_argument("-end_required", default="0", type=lambda v: str(v).strip().lower() in ("1", "true", "yes", "on"),
                        help="1 (or true): the end of the walk must be recorded (sensors still at the end), as for "
                             "the analysis of the last steps; a recording that ends while walking is then critical")
    args = parser.parse_args()
    out = os.getcwd()

    exists = lambda p: p is not None and os.path.isfile(p) and os.path.getsize(p) > 0
    files = {"HE": args.i0 if exists(args.i0) else None, "LB": args.i1, "LF": args.i2, "RF": args.i3}
    with open(os.path.join(out, "algo_info.txt"), "w") as f:
        f.write("head = {}\n".format(int(files["HE"] is not None)))
    try:
        missing = [name for name, path in (("lower back", args.i1), ("left foot", args.i2), ("right foot", args.i3))
                   if not exists(path)]
        if missing:
            raise ValueError("raw file missing for: {}".format(", ".join(missing)))
        segmentation = args.i4 if exists(args.i4) else None
        params = default_params()
        params["settings"]["end_standing_required"] = bool(args.end_required)
        a = assess_files(files, segmentation, args.freq, trial_id="uploaded trial", params=params,
                         acc_unit=args.acc_unit, gyr_unit=args.gyr_unit)
    except Exception as error:
        with open(os.path.join(out, "demo_failure.txt"), "w") as f:
            f.write("The trial could not be scored: {}\n".format(error))
        return 0
    write_reports(a, out)
    save_figures(a, out)
    for s, p in (a.placement or {}).items():
        if files.get(s):
            write_corrected(files[s], p, os.path.join(out, "corrected_{}.txt".format(s)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
