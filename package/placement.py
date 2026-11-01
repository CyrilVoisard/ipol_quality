# Copyright (C) 2026 Cyril Voisard
# SPDX-License-Identifier: AGPL-3.0-or-later
# Detection and correction of the placement of the sensors (indicators.md, section 4).

from dataclasses import dataclass

import numpy as np

QUARTERS = (0, 90, 180, 270)


def rz(deg):
    """Rotation by deg about Z. A sensor turned by deg in its plane measures the classic vectors
    rotated by -deg; rz(deg) brings them back: a_classic = rz(deg) @ a_raw."""
    t = np.radians(deg)
    c, s = int(round(np.cos(t))), int(round(np.sin(t)))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


@dataclass
class Placement:
    sensor: str
    rotation: int            # rotation of the sensor in its plane, degrees about Z (0 = classic)
    R: np.ndarray            # a_classic = R @ a_raw
    in_plane: float          # share of the gravity in the XY plane of the sensor (0..1)
    margin: float            # angular distance to the nearest quarter turn boundary (deg, 0..45)
    decided_by: str          # "gravity" or "walk"

    @property
    def classic(self):
        return self.rotation == 0

    def describe(self, lang="fr"):
        if self.classic:
            return "position classique" if lang == "fr" else "classic placement"
        angle = {90: "+90°", 180: "180°", 270: "−90°"}[self.rotation]
        if lang == "fr":
            return "variante : tourné de {} dans son plan (corrigé)".format(angle)
        return "variant: rotated by {} in its plane (corrected)".format(angle)

    def mapping(self):
        """e.g. "-Y,+X,+Z": classic X = -raw Y, classic Y = +raw X, classic Z = +raw Z."""
        out = []
        for i in range(3):
            j = int(np.argmax(np.abs(self.R[i])))
            out.append("{}{}".format("+" if self.R[i, j] > 0 else "-", "XYZ"[j]))
        return ",".join(out)


def detect(aligned, stillness, settings):
    """Placement of every sensor provided (dictionary sensor -> Placement)."""
    from .indicators.sensor import rest_window
    out = {}
    walk = stillness.active.slice() if stillness.active.end - stillness.active.start >= 2 else slice(0, aligned.n)
    for s in aligned.sensors:
        g = aligned.acc[s][rest_window(aligned, stillness, s, settings)].mean(axis=0)
        gxy = np.hypot(g[0], g[1])
        in_plane = float(gxy / (np.linalg.norm(g) + 1e-12))
        phi = np.degrees(np.arctan2(g[1], g[0]))
        k = int(np.round(phi / 90.0)) % 4          # gravity at k quarter turns -> sensor turned by -k
        margin = 45.0 - abs(((phi - 90.0 * np.round(phi / 90.0)) + 180.0) % 360.0 - 180.0)
        decided = "gravity"
        if s in ("LF", "RF"):
            # feet: the gravity at rest leans on the dorsum and can point anywhere near the boundary
            # between two quarter turns; the medio-lateral axis (Y) is the axis of the main angular
            # velocity during the walk, the gravity only gives the sign (classic X carries a
            # positive share of it)
            e = np.mean(aligned.gyr[s][walk] ** 2, axis=0)
            k = (0 if g[0] >= 0 else 2) if e[1] >= e[0] else (1 if g[1] >= 0 else 3)   # same convention
            decided = "walk"
        rotation = QUARTERS[(-k) % 4]
        out[s] = Placement(s, rotation, rz(rotation), in_plane, float(margin), decided)
    return out


def apply(aligned, placements):
    """Rotate the signals of the aligned trial into the classic frames (in place)."""
    for s, p in placements.items():
        if not p.classic:
            aligned.acc[s] = aligned.acc[s] @ p.R.T
            aligned.gyr[s] = aligned.gyr[s] @ p.R.T
    return aligned
