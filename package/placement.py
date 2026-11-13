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


def angle_to_x(g):
    """Angle (deg) between the vector g and +X (the tilt of a head or lower-back sensor)."""
    return float(np.degrees(np.arccos(np.clip(g[0] / (np.linalg.norm(g) + 1e-12), -1.0, 1.0))))


def tilt_rotation(g):
    """Smallest rotation taking the direction of g onto +X (Rodrigues): R @ g / |g| = (1, 0, 0).
    Only the tilt is corrected; the rotation about the vertical (heading) is left as it is."""
    a = np.asarray(g, float) / (np.linalg.norm(g) + 1e-12)
    v = np.cross(a, [1.0, 0.0, 0.0])
    s, c = np.linalg.norm(v), a[0]
    if s < 1e-9:
        return np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / s ** 2)


@dataclass
class Placement:
    sensor: str
    rotation: int            # rotation of the sensor in its plane, degrees about Z (0 = classic)
    R: np.ndarray            # a_classic = R @ a_raw
    in_plane: float          # share of the gravity in the XY plane of the sensor (0..1)
    margin: float            # angular distance to the nearest quarter turn boundary (deg, 0..45)
    decided_by: str          # "gravity" or "walk"
    # alignment on the vertical, written in the corrected file only (settings["tilt_sensors"])
    R_tilt: np.ndarray = None        # a_corrected = R_tilt @ R @ a_raw (None: no alignment)
    tilt_rest: float = None          # tilt at rest (deg, classic frame), None without standing period
    tilt_walk: float = None          # tilt of the mean acceleration over the walk (deg)
    tilt_reference: str = None       # vertical used: "walk" or "rest" (or why it was not aligned)

    @property
    def classic(self):
        return self.rotation == 0

    @property
    def R_corrected(self):
        """Rotation written in corrected_XX.txt: quarter turn, then alignment on the vertical."""
        return self.R if self.R_tilt is None else self.R_tilt @ self.R

    @property
    def tilt_corrected(self):
        """Angle (deg) of the alignment on the vertical (0 when none)."""
        if self.R_tilt is None:
            return 0.0
        return float(np.degrees(np.arccos(np.clip((np.trace(self.R_tilt) - 1.0) / 2.0, -1.0, 1.0))))

    def describe_tilt(self):
        if self.R_tilt is None:
            return None if self.tilt_reference is None else "not aligned on the vertical ({})".format(self.tilt_reference)
        rest = "" if self.tilt_rest is None else ", {:.1f}° at rest".format(self.tilt_rest)
        return "tilt {:.1f}° during the walk{}: X aligned on the vertical ({}) in corrected_{}.txt".format(
            self.tilt_walk, rest, self.tilt_reference, self.sensor)

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
        p = Placement(s, rotation, rz(rotation), in_plane, float(margin), decided)
        if s in settings.get("tilt_sensors", ()):
            _tilt(p, aligned, stillness, walk, g, settings)
        out[s] = p
    return out


def _tilt(p, aligned, stillness, walk, g_rest, settings):
    """Alignment of the X axis of sensor p.sensor on the vertical (classic frame, after the quarter
    turn). The rest value is not used without standing period (the rest window is then taken while
    walking); without a walk the rest value is the only one."""
    has_rest = stillness.rest.label != "quietest"
    has_walk = stillness.active.end - stillness.active.start >= 2
    g_rest = p.R @ g_rest if has_rest else None
    g_walk = p.R @ aligned.acc[p.sensor][walk].mean(axis=0) if has_walk else None
    p.tilt_rest = None if g_rest is None else angle_to_x(g_rest)
    p.tilt_walk = None if g_walk is None else angle_to_x(g_walk)
    reference = settings.get("tilt_reference", "walk")
    g = g_walk if reference == "walk" else g_rest
    if g is None:
        reference, g = ("rest", g_rest) if g_rest is not None else ("walk", g_walk)
    if g is None:
        p.tilt_reference = "no rest nor walk"
    elif g[0] <= 0:
        # gravity not on +X: sensor on another face or axes swapped, scored by orient_rest (gate)
        p.tilt_reference = "gravity not on +X"
    elif angle_to_x(g) > settings.get("tilt_max_deg", 75.0):
        # beyond a plausible tilt of the trunk: sensor on another face, scored by orient_rest
        p.tilt_reference = "tilt above {:g}°".format(settings.get("tilt_max_deg", 75.0))
    else:
        p.R_tilt, p.tilt_reference = tilt_rotation(g), reference


def write_corrected(path_in, placement, path_out):
    """Copy of a raw sensor file with its Acc_*, Gyr_* (and Mag_*) columns turned into the classic
    frame and, for the sensors of settings["tilt_sensors"], with X aligned on the vertical
    (placement.R_corrected), so that the other demos (U-turn, gait events) and the gait parameters
    can be run on corrected axes. The other columns, the samples and the header lines are kept as
    they are."""
    import pandas as pd
    from .io import header_row
    with open(path_in, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()
    skip = header_row(path_in)
    data = pd.read_csv(path_in, sep=r"[,\t; ]+", skiprows=skip, engine="python", encoding="utf-8-sig",
                       encoding_errors="replace")
    data.columns = [str(c).strip() for c in data.columns]
    for kind in ("Acc", "Gyr", "Mag"):
        cols = ["{}_{}".format(kind, k) for k in "XYZ"]
        if all(c in data.columns for c in cols):
            values = data[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            data[cols] = values @ placement.R_corrected.T
    with open(path_out, "w") as f:
        f.writelines(lines[:skip])
        data.to_csv(f, sep="\t", index=False, float_format="%.9g", lineterminator="\n")


def apply(aligned, placements):
    """Rotate the signals of the aligned trial into the classic frames (in place)."""
    for s, p in placements.items():
        if not p.classic:
            aligned.acc[s] = aligned.acc[s] @ p.R.T
            aligned.gyr[s] = aligned.gyr[s] @ p.R.T
    return aligned
