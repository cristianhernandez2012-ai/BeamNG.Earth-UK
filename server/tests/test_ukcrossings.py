"""Pedestrian crossings OSM maps, painted as in the British Isles (ukcrossings.py)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roads  # noqa: E402
import ukcrossings as uk  # noqa: E402


def road(cls="tertiary", oneway=False, d=0):
    pts = np.array([(-100.0, 0.0, 5.0), (100.0, 0.0, 5.0)])
    return ("r", roads.Spec(8.0, "paved", True, cls, oneway, dir=d), pts, 0.0)


class KindTest(unittest.TestCase):
    def test_kinds_from_the_tags(self):
        self.assertEqual(uk.kind_of({"crossing": "uncontrolled", "crossing:markings": "zebra"}), uk.ZEBRA)
        self.assertEqual(uk.kind_of({"crossing": "zebra"}), uk.ZEBRA)
        self.assertEqual(uk.kind_of({"crossing_ref": "zebra"}), uk.ZEBRA)
        self.assertEqual(uk.kind_of({"crossing": "traffic_signals", "crossing_ref": "toucan"}), uk.SIGNAL)
        self.assertEqual(uk.kind_of({"crossing_ref": "pelican"}), uk.SIGNAL)
        self.assertIsNone(uk.kind_of({"crossing": "unmarked", "crossing:markings": "no"}))
        self.assertIsNone(uk.kind_of({}))


class MarksTest(unittest.TestCase):
    def test_a_zebra_has_stripes_give_way_lines_and_zig_zags(self):
        out = uk.marks([road()], np.array([[0.0, 1.0, uk.ZEBRA]]))
        stripes = [m for m in out if m[0] == "zebra"]
        self.assertEqual(len(stripes), 1)
        self.assertTrue(np.allclose(stripes[0][1][:, 0], 0.0), "across the road at the node")
        self.assertAlmostEqual(float(stripes[0][1][0, 2]), 5.0, msg="on the road")
        giveway = [m for m in out if m[2] == uk.GIVEWAY_W]
        self.assertEqual(sorted(round(float(m[1][0, 0]), 2) for m in giveway),
                         [-(uk.ZEBRA_W / 2 + uk.GIVEWAY_BACK_M + uk.GIVEWAY_W / 2),
                          uk.ZEBRA_W / 2 + uk.GIVEWAY_BACK_M + uk.GIVEWAY_W / 2])
        zig = [m for m in out if m[2] == uk.ZIGZAG_W]
        self.assertEqual(len(zig), 2 * 3 * 2 * uk.ZIGZAG_MARKS, "two approaches, two kerbs and the centre")
        xs = np.concatenate([m[1][:, 0] for m in zig])
        self.assertLessEqual(float(np.abs(xs).max()), 3.4 + uk.ZIGZAG_MARKS * uk.ZIGZAG_MARK_M + 0.01)

    def test_a_signalled_crossing_has_studs_and_stop_lines_on_the_incoming_halves(self):
        for left_hand in (True, False):
            out = uk.marks([road()], np.array([[0.0, 0.0, uk.SIGNAL]]), left_hand=left_hand)
            self.assertFalse([m for m in out if m[0] == "zebra"])
            self.assertEqual(len([m for m in out if m[2] == uk.STUD_W]), 2)
            stops = [m for m in out if m[2] == uk.STOP_W]
            self.assertEqual(len(stops), 2)
            for _l, ln, _w, _d in stops:
                x, y = float(ln[0, 0]), float(np.mean(ln[:, 1]))
                # traffic from the east (x > 0) heads west: its left is south (y < 0)
                west_bound = x > 0
                self.assertEqual(y < 0, west_bound == left_hand, (left_hand, x, y))

    def test_one_way_and_unmarked(self):
        out = uk.marks([road(oneway=True, d=1)], np.array([[0.0, 0.0, uk.SIGNAL]]), left_hand=True)
        self.assertEqual(len([m for m in out if m[2] == uk.STOP_W]), 1, "a stop line for the traffic that comes")
        self.assertEqual(len([m for m in out if m[2] == uk.ZIGZAG_W]), 2 * 2 * 2 * uk.ZIGZAG_MARKS, "no centre")
        self.assertEqual(uk.marks([road()], np.array([[0.0, 20.0, uk.ZEBRA]])), [], "not on the road")
        self.assertEqual(uk.marks([road()], None), [])

    def test_build_tile_paints_them_in_the_british_isles_only(self):
        from test_roads import FRAME, features, flat
        import junctions
        f = features([[(-100, 50), (300, 50)]], ["tertiary"])
        cr = np.array([[100.0, 50.0, uk.ZEBRA]])
        urban = lambda la, lo: np.ones(np.shape(la), bool)    # noqa: E731
        gb = roads.build_tile(FRAME, 0, 0, [f], flat, urban, walks=[], left_hand=True, country="GB", crossings=cr)
        us = roads.build_tile(FRAME, 0, 0, [f], flat, urban, walks=[], country="US", crossings=cr)
        self.assertTrue([c for c in gb if c["m"] == junctions.CROSSWALK["m"]])
        self.assertTrue([c for c in gb if c["w"] == uk.ZIGZAG_W])
        self.assertFalse([c for c in us if c["w"] == uk.ZIGZAG_W])


if __name__ == "__main__":
    unittest.main()
