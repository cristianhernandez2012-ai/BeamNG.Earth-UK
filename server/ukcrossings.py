"""Pedestrian crossings OpenStreetMap maps, painted as the British Isles paint them (markings.Style.junction_zebra
False: GB / IE / IM / JE / GG), where a junction carries no crosswalk of its own (the user, Basildon 2026-11-02: the
road markings OSM knows of were missing; Basildon maps ~136 zebras and ~116 signalled crossings, Overture infrastructure
class "crossing" with the node's tags).

The Traffic Signs Regulations and General Directions 2016, Schedule 14 (pedestrian crossings):
- a zebra (crossing=zebra, crossing_ref=zebra, crossing:markings=zebra): the black and white stripes across the
  carriageway (ZEBRA_W m along it), a give-way line on each side (diagram 1001.2: a broken line, 200 mm; 1 m back);
- a signal-controlled crossing (crossing=traffic_signals; pelican, puffin, toucan, pegasus): two lines of studs at the
  crossing's edges (STUD_W, its width SIGNAL_W m), a stop line on each approach's incoming half (diagram 1001,
  STOP_BACK_M back from the studs);
- both: the zig-zag controlled area on each approach (diagram 1001.3 / 1001.4: ZIGZAG_MARKS marks of ZIGZAG_MARK_M
  along each kerb and, on a two-way road, the centre), as far as the road part reaches;
- an unmarked or unknown crossing: nothing.
The marks follow the road's line (its own curve) through the crossing node; z the road's. Pure functions of the road
lines and the crossing points.
"""
import json
import math

import numpy as np

ZEBRA, SIGNAL = 1, 2
SIGNAL_REFS = {"pelican", "puffin", "toucan", "pegasus", "equestrian", "traffic_signals"}
ZEBRA_W = 3.0               # m of stripes along the road (2.4-5 m)
SIGNAL_W = 3.0              # m between the stud lines (2.4 m at least)
STUD_W = 0.15               # m: a stud line's width (its marks STUD_DASH)
STUD_DASH = (0.15, 0.25)
GIVEWAY_BACK_M = 1.0        # m from the stripes to the zebra's give-way line
GIVEWAY_W = 0.2
GIVEWAY_DASH = (0.5, 0.5)
STOP_BACK_M = 1.7           # m from the studs to the stop line (1.7 m at least)
STOP_W = 0.3
ZIGZAG_MARKS = 8            # marks per approach (2 to 18)
ZIGZAG_MARK_M = 2.0         # m along the road per mark
ZIGZAG_AMP = 0.25           # m across: a mark's zig
ZIGZAG_W = 0.1
ZIGZAG_GAP_M = 0.5          # m from the give-way / stop line to the zig-zags
KERB_IN_M = 0.3             # m inside the kerb the kerb zig-zags run
SNAP_M = 4.0                # m: a crossing node this far off a road's line is not on it
DRIVE_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
                 "living_street"}


def kind_of(tags):
    """ZEBRA, SIGNAL or None of an OSM crossing node's tags (dict)."""
    if not tags:
        return None
    crossing = str(tags.get("crossing") or "").lower()
    ref = str(tags.get("crossing_ref") or "").lower()
    marks = str(tags.get("crossing:markings") or "").lower()
    if crossing == "traffic_signals" or ref in SIGNAL_REFS or crossing in SIGNAL_REFS:
        return SIGNAL
    if crossing == "zebra" or ref == "zebra" or "zebra" in marks.split(";"):
        return ZEBRA
    return None


def mapped(frame, infra_list):
    """(n, 3) frame [x, y, kind] of the crossings in infra_list (Overture infrastructure Features, class "crossing" /
    "pedestrian_crossing", their source tags), marked ones only; duplicates across cells once."""
    seen, rows = set(), []
    for f in infra_list or ():
        if f is None or not len(f):
            continue
        cls, tags = f.attrs.get("class"), f.attrs.get("tags")
        if cls is None or tags is None:
            continue
        hit = np.isin(np.asarray(cls).astype(str), ["crossing", "pedestrian_crossing"])
        for i in np.flatnonzero(hit):
            fid = str(f.ids[i])
            if fid in seen:
                continue
            seen.add(fid)
            try:
                t = json.loads(str(tags[i])) if tags[i] is not None else None
            except (TypeError, ValueError):
                t = None
            k = kind_of(t if isinstance(t, dict) else None)
            if k is None:
                continue
            a = f.coord_off[f.ring_off[f.part_off[i]]]
            b = f.coord_off[f.ring_off[f.part_off[i + 1]]]
            lon, lat = f.lonlat[a:b].mean(axis=0)
            rows.append((lat, lon, k))
    if not rows:
        return np.zeros((0, 3))
    r = np.asarray(rows, np.float64)
    e, n, _ = frame.geodetic_to_enu(r[:, 0], r[:, 1], 0.0)
    return np.column_stack([np.asarray(e), np.asarray(n), r[:, 2]])


def crossings_for(md, frame, keys):
    """mapped() of the map cells keys; None when their infrastructure cannot be read."""
    try:
        return mapped(frame, [md.cell(k, layers=("infrastructure",), kind="demand").layers.get("infrastructure")
                              for k in keys])
    except Exception as e:      # no crossings rather than no road tile
        import logging
        logging.getLogger("earthd.ukcrossings").debug("crossings: %s: %s", type(e).__name__, e)
        return None


def _left(u):
    return np.array([-u[1], u[0]])


class _Line:
    """A road's centre line (k, 3) by arc length: point, tangent."""

    def __init__(self, pts):
        self.p = np.asarray(pts, np.float64)
        self.s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(self.p[:, :2], axis=0).T))]
        self.L = float(self.s[-1])

    def at(self, t):
        t = min(max(t, 0.0), self.L)
        j = int(min(max(np.searchsorted(self.s, t, side="right") - 1, 0), len(self.p) - 2))
        a, b = self.p[j], self.p[j + 1]
        seg = self.s[j + 1] - self.s[j]
        f = (t - self.s[j]) / seg if seg > 1e-9 else 0.0
        q = a + (b - a) * f
        d = (b - a)[:2]
        n = float(np.hypot(*d))
        return q, (d / n if n > 1e-9 else np.array([1.0, 0.0]))

    def project(self, x, y):
        """(arc m, distance m) of the nearest point."""
        a, b = self.p[:-1, :2], self.p[1:, :2]
        d = b - a
        L2 = np.maximum((d * d).sum(1), 1e-12)
        t = np.clip(((np.array([x, y]) - a) * d).sum(1) / L2, 0.0, 1.0)
        q = a + t[:, None] * d
        dist = np.hypot(q[:, 0] - x, q[:, 1] - y)
        j = int(np.argmin(dist))
        return float(self.s[j] + t[j] * (self.s[j + 1] - self.s[j])), float(dist[j])


def _across(line, t, lo, hi, z_lift=0.0):
    """(2, 3) a straight line across the road at arc t, from lateral offset lo to hi (+ left of the line)."""
    q, u = line.at(t)
    n = _left(u)
    a, b = q[:2] + n * lo, q[:2] + n * hi
    z = float(q[2]) + z_lift if len(q) > 2 else z_lift
    return np.array([[a[0], a[1], z], [b[0], b[1], z]])


def marks(roads, crossings, box=None, left_hand=False, owns=None):
    """[(look, (k, 3) line, width, dashes or None)] of the crossings (mapped()'s rows) on roads [(fid, spec, pts
    (k, 3), lon)]: look "zebra" (the stripes: a crosswalk chunk), "line" (white lines). owns(x, y): the tile's own
    crossings (default: all)."""
    out = []
    if crossings is None or not len(crossings):
        return out
    lines = [(spec, _Line(pts)) for _fid, spec, pts, _lon in roads
             if spec is not None and getattr(spec, "cls", None) in DRIVE_CLASSES and len(pts) >= 2
             and getattr(spec, "kind", "paved") == "paved"]
    if not lines:
        return out
    for x, y, k in crossings:
        if owns is not None and not owns(x, y):
            continue
        best = None
        for spec, ln in lines:
            t, d = ln.project(x, y)
            if d <= SNAP_M and (best is None or d < best[0]):
                best = (d, spec, ln, t)
        if best is None:
            continue
        _d, spec, ln, t = best
        half = spec.width / 2
        if half < 1.5:
            continue
        one_way = bool(spec.oneway) and bool(getattr(spec, "dir", 0))
        full = (-half + 0.1, half - 0.1)
        if int(k) == ZEBRA:
            out.append(("zebra", _across(ln, t, *full), ZEBRA_W, None))
            reach = ZEBRA_W / 2
            for sgn in (1, -1):
                tt = t + sgn * (reach + GIVEWAY_BACK_M + GIVEWAY_W / 2)
                out.append(("line", _across(ln, tt, *full), GIVEWAY_W, GIVEWAY_DASH))
            back = reach + GIVEWAY_BACK_M + GIVEWAY_W
        else:
            reach = SIGNAL_W / 2
            for sgn in (1, -1):
                out.append(("line", _across(ln, t + sgn * reach, *full), STUD_W, STUD_DASH))
            for sgn in (1, -1):         # the approach from the +u side (sgn 1) travels -u, from the -u side +u
                travel_dir = -sgn       # along the line: -1 / +1
                if one_way and int(getattr(spec, "dir", 0)) != travel_dir:
                    continue            # (no traffic comes from this side)
                tt = t + sgn * (reach + STOP_BACK_M + STOP_W / 2)
                if one_way:
                    lo, hi = full
                else:                   # the incoming half: left of travel when driving on the left
                    keep_left = (travel_dir > 0) == left_hand
                    lo, hi = (0.1, half - 0.2) if keep_left else (-half + 0.2, -0.1)
                out.append(("line", _across(ln, tt, lo, hi), STOP_W, None))
            back = reach + STOP_BACK_M + STOP_W
        # the zig-zags on each approach, along each kerb and (two-way) the centre
        offs = [half - KERB_IN_M, -(half - KERB_IN_M)] + ([] if one_way else [0.0])
        for sgn in (1, -1):
            start = t + sgn * (back + ZIGZAG_GAP_M)
            for o in offs:
                inward = -np.sign(o) if o else 1.0
                pts = []
                for m in range(2 * ZIGZAG_MARKS + 1):
                    tt = start + sgn * m * ZIGZAG_MARK_M / 2
                    if not (0.0 <= tt <= ln.L):
                        break
                    q, u = ln.at(tt)
                    lat = o + (inward * ZIGZAG_AMP if m % 2 else 0.0)
                    p2 = q[:2] + _left(u) * lat
                    pts.append([p2[0], p2[1], float(q[2]) if len(q) > 2 else 0.0])
                for a, b in zip(pts[:-1], pts[1:]):
                    out.append(("line", np.array([a, b]), ZIGZAG_W, None))
    return out
