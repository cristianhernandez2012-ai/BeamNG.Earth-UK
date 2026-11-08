"""The British Isles' junction and lane markings drawn as white strokes: the hatching where a road splits into a
one-way pair (TSRGD 2016 diagram 1040), lane arrows from OSM's turn:lanes (diagram 1038), the BUS LANE legend (1048)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import furniture  # noqa: E402
import markings  # noqa: E402
import roads  # noqa: E402


BOX = (-500.0, -500.0, 500.0, 500.0)


def spec(cls, width, oneway=False, d=0):
    return roads.Spec(width, "paved", True, cls, oneway, dir=d)


class HatchTest(unittest.TestCase):
    def split(self):
        """A two-way road from the west to (0, 0), splitting into an exit (north-east, one way out) and an entry
        (south-east, one way in): a roundabout approach's splitter island."""
        ang = np.radians(20)
        out_ = np.array([[0, 0, 0], [60 * np.cos(ang), 60 * np.sin(ang), 0]], float)
        in_ = np.array([[60 * np.cos(ang), -60 * np.sin(ang), 0], [0, 0, 0]], float)
        west = np.array([[-100, 0, 0], [0, 0, 0]], float)
        return [(1, spec("tertiary", 7.0), west, 0.0), (2, spec("tertiary", 4.0, True, 1), out_, 0.0),
                (3, spec("tertiary", 4.0, True, 1), in_, 0.0)]

    def test_between_the_pair(self):
        rows = [r for r in furniture.junction_markings(self.split(), BOX, True, country="GB") if r[0] == furniture.HATCH]
        self.assertGreaterEqual(len(rows), 4, "two boundary lines and stripes")
        pts = np.concatenate([r[1] for r in rows])
        self.assertTrue((pts[:, 0] > 0).all(), "past the split, between the pair")
        for p in pts:
            a = np.degrees(np.arctan2(p[1], p[0]))
            self.assertLess(abs(a), 20.0 + 1e-6, "inside the wedge")

    def test_none_where_no_split(self):
        rs = self.split()
        rs[2] = (3, spec("tertiary", 4.0, True, -1), rs[2][2], 0.0)      # both one way out
        self.assertFalse([r for r in furniture.junction_markings(rs, BOX, True, country="GB") if r[0] == furniture.HATCH])
        self.assertFalse([r for r in furniture.junction_markings(self.split(), BOX, True, country="DE")
                          if r[0] == furniture.HATCH], "elsewhere none")


class ArrowTest(unittest.TestCase):
    def cross(self):
        X = np.array([0.0, 0.0, 0.0])
        return [(1, spec("primary", 10.0), np.array([[-100, 0, 0], X]), 0.0),
                (2, spec("primary", 10.0), np.array([X, [100, 0, 0]]), 0.0),
                (3, spec("residential", 7.0), np.array([X, [0, 100, 0]]), 0.0)]

    def test_the_map_s_lanes_only(self):
        tags = {1: {"turn:lanes:forward": "left|through;right"}}
        rows = [r for r in furniture.junction_markings(self.cross(), BOX, True, country="GB", tags=tags)
                if r[0] == furniture.ARROW]
        self.assertTrue(rows)
        pts = np.concatenate([r[1] for r in rows])
        self.assertTrue((pts[:, 0] < 0).all(), "on the west approach only")
        self.assertTrue((pts[:, 1] > 0).all(), "in the incoming (left) half when driving on the left")
        self.assertFalse([r for r in furniture.junction_markings(self.cross(), BOX, True, country="GB", tags={})
                          if r[0] == furniture.ARROW], "nothing guessed")

    def test_lane_turns(self):
        arm = (1, spec("primary", 10.0), np.array([1.0, 0.0]), np.zeros(3), False, 100.0)
        self.assertEqual(furniture._lane_turns({"turn:lanes:forward": "left|through;right"}, arm), [{-1}, {0, 1}])
        self.assertIsNone(furniture._lane_turns({"turn:lanes:forward": "none|none"}, arm))
        self.assertIsNone(furniture._lane_turns({"turn:lanes:backward": "left|through"}, arm))


class LegendTest(unittest.TestCase):
    def test_bus_lane_legend_in_the_bus_lane(self):
        sp = roads.Spec(9.9, "paved", True, "primary", True, dir=1)
        pts = np.array([(0.0, 0.0, 0.0), (200.0, 0.0, 0.0)])
        ex = roads.extras(sp, pts, 0.0, True, "GB", {"bus:lanes": "designated||", "lanes": "3"})
        leg = [e for e in ex if e[0].get("kind") == "legend"]
        self.assertTrue(leg)
        p = np.concatenate([e[2] for e in leg])
        self.assertTrue((p[:, 1] > 1.65).all() and (p[:, 1] < 4.95).all(), "in the kerbside (left) lane")
        first = p[p[:, 0] < 100]
        self.assertAlmostEqual(float(first[:, 0].min()), markings.LEGEND_FIRST_M, delta=0.01)
        self.assertFalse([e for e in roads.extras(sp, pts, 0.0, False, "DE", {"bus:lanes": "designated||",
                                                                              "lanes": "3"})
                          if e[0].get("kind") == "legend"])

    def test_reads_towards_the_driver(self):
        strokes = markings.legend(np.array([(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]), 0.0, -1, words=("L",))
        ys = np.concatenate(strokes)
        # travel -x: the L's foot (its bar) nearest the driver: at the largest x of the letter
        bar = [s for s in strokes if np.ptp(s[:, 1]) > 0.4]
        self.assertTrue(bar)
        self.assertGreater(float(ys[:, 0].max()), 80.0)


if __name__ == "__main__":
    unittest.main()
