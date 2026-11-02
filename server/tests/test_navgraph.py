"""navgraph: the drivable roads as the stock map's graph: connectors shared at junctions, one-way direction,
lanes, speed limits, clipped to the window."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import navgraph  # noqa: E402
import roads  # noqa: E402
from test_roads import FRAME, features, flat, rules  # noqa: E402


def graph(lines, classes, connectors=(), oneway=(), speeds=(), subclass=(), centre=(0.0, 0.0), window=1100.0):
    f = features(lines, classes)
    f.rules["connectors"] = rules(list(connectors))
    f.rules["oneway"] = rules(list(oneway), "float")
    f.rules["max_speed_kmh"] = rules(list(speeds), "float")
    f.rules["subclass"] = rules(list(subclass))
    return navgraph.build(FRAME, [f], flat, centre, window=window)


class GraphTest(unittest.TestCase):
    def test_junction_shares_the_connector(self):
        g = graph([[(0, 0), (100, 0)], [(100, 0), (100, 100)]], ["residential", "residential"],
                  connectors=[(0, "A", 0.0, 0.0), (0, "J", 1.0, 1.0), (1, "J", 0.0, 0.0), (1, "B", 1.0, 1.0)])
        self.assertIn("cJ", g["nodes"])
        links = [e for e in g["edges"] if "cJ" in e[:2]]
        self.assertEqual(len(links), 2, "both roads end at the junction node")
        x, y, z, r = g["nodes"]["cJ"]
        self.assertAlmostEqual(x, 100.0, delta=0.5)
        self.assertAlmostEqual(y, 0.0, delta=0.5)
        self.assertEqual(z, 10.0)
        for e in g["edges"]:
            self.assertEqual((e[3], e[4]), (1, 1), "a two-way residential street: one lane each way")
            self.assertAlmostEqual(e[5], 25 * navgraph.MPH, places=2)

    def test_offset_connector_at_the_widest_road(self):
        """A ramp's line ends 3.3 m off the motorway's joint at their shared connector (I-83 in Baltimore): the node
        stands on the motorway's line, whichever road comes first."""
        for order in (0, 1):
            lines = [[(0, -200), (0, 0)], [(0, 0), (0, 200)], [(60, -150), (3.3, 0)]]
            classes = ["primary", "primary", "residential"]
            con = [(0, "J", 1.0, 1.0), (1, "J", 0.0, 0.0), (2, "J", 1.0, 1.0)]
            if order:
                lines, classes = lines[::-1], classes[::-1]
                con = [(2 - f, c, a, b) for f, c, a, b in con]
            x, y, _z, _r = graph(lines, classes, connectors=con)["nodes"]["cJ"]
            self.assertAlmostEqual(x, 0.0, delta=0.05, msg=order)
            self.assertAlmostEqual(y, 0.0, delta=0.05)

    def test_long_edges_split_and_oneway_against_the_geometry(self):
        g = graph([[(0, 0), (300, 0)]], ["secondary"], connectors=[(0, "A", 0.0, 0.0), (0, "B", 1.0, 1.0)],
                  oneway=[(0, -1.0)], speeds=[(0, 72.0)])
        self.assertGreaterEqual(len(g["edges"]), 4, "300 m in edges of at most MAX_EDGE_M")
        for e in g["edges"]:
            self.assertEqual(e[2], 1)
            self.assertLess(g["nodes"][e[1]][0], g["nodes"][e[0]][0], "one-way -1: drives against the geometry")
            self.assertEqual((e[3], e[4]), (e[3], 0))
            self.assertAlmostEqual(e[5], 20.0, places=2)
        self.assertTrue(any(e[1] == "cA" for e in g["edges"]))

    def test_not_drivable_and_window(self):
        g = graph([[(0, 0), (50, 0)], [(0, 10), (50, 10)], [(0, 20), (50, 20)], [(5000, 0), (5100, 0)]],
                  ["footway", "service", "residential", "residential"], subclass=[(1, "parking_aisle")])
        ys = {round(n[1]) for n in g["nodes"].values()}
        self.assertEqual(ys, {20}, "only the residential street in the window")

    def test_even_lanes_and_no_parking_lanes(self):
        import roads
        for cls in ("residential", "tertiary", "secondary", "primary"):
            sp = roads.Spec(roads.carriageway(cls, "paved", True), "paved", True, cls, False)
            w = navgraph.driving_width(sp)
            self.assertLessEqual(w, sp.width)
            lo, li = navgraph._lanes(w, False)
            self.assertEqual(lo, li, cls)

    def test_rural_motorway_two_lanes(self):
        import roads
        for urban, n in ((False, 2), (True, 3)):
            sp = roads.Spec(roads.carriageway("motorway", "paved", urban, None, True), "paved", urban, "motorway", True)
            self.assertEqual(navgraph._lanes(navgraph.driving_width(sp), True), (n, 0), urban)
        ramp = roads.Spec(roads.LINK_M[1], "paved", False, "motorway", True)
        self.assertEqual(navgraph._lanes(navgraph.driving_width(ramp), True), (1, 0), "a ramp: one lane (GitHub #107)")

    def test_nodes_on_the_curve_of_the_earth(self):
        """10 km from the anchor the ground drawn lies ~7.8 m under the flat plane (the anchor frame's up): the nodes
        too, not at the raw height (the AI's nodes and traffic lights stood over the ground, PR #112's finding)."""
        g = graph([[(0, 10000), (200, 10000)]], ["primary"], centre=(100.0, 10000.0))
        zs = np.array([n[2] for n in g["nodes"].values()])
        self.assertTrue(len(zs))
        drop = 10000.0 ** 2 / (2 * 6371000.0)
        np.testing.assert_allclose(zs, 10.0 - drop, atol=0.15)

    def test_tunnel_stretch_left_out(self):
        f = features([[(0, 0), (300, 0)]], ["primary"], levels=[(0, -1.0, 0.4, 0.7)])
        f.rules["connectors"] = rules([(0, "A", 0.0, 0.0), (0, "B", 1.0, 1.0)])
        g = navgraph.build(FRAME, [f], flat, (0.0, 0.0))
        for e in g["edges"]:
            mid = (g["nodes"][e[0]][0] + g["nodes"][e[1]][0]) / 2
            self.assertFalse(130 < mid < 200, "an edge over the tunnel")
        self.assertTrue(g["edges"])

    def test_nodes_on_a_bridge_take_its_deck(self):
        f = features([[(0, 0), (200, 0)]], ["primary"])
        f.rules["connectors"] = rules([(0, "A", 0.0, 0.0), (0, "B", 1.0, 1.0)])
        decks = {"id000": [(np.array([(50.0, 0.0), (150.0, 0.0)]), np.array([10.0, 14.0]))]}
        g = navgraph.build(FRAME, [f], flat, (0.0, 0.0), decks=decks)
        on = [n for n in g["nodes"].values() if 55 < n[0] < 145]
        off = [n for n in g["nodes"].values() if n[0] < 40 or n[0] > 160]
        self.assertTrue(on and off)
        for n in on:
            self.assertAlmostEqual(n[2], 10.0 + 4.0 * (n[0] - 50.0) / 100.0, delta=0.05)
        self.assertTrue(all(n[2] == 10.0 for n in off), "the ground (flat 10 m) off the bridge")



class SignalsTest(unittest.TestCase):
    def test_no_stop_on_a_link_inside_one_intersection(self):
        import roads
        sp = roads.Spec(8.0, "paved", True, "residential", False)
        nodes = {"cJ": [0.0, 0.0, 0.0, 4], "cK": [0.0, 8.0, 0.0, 4], "s1": [0.0, -40.0, 0.0, 2],
                 "cW": [-40.0, 0.0, 0.0, 2], "cE": [40.0, 0.0, 0.0, 2]}
        arms = {"cJ": [("n", sp, np.array([0.0, 1.0]), 0.0, 1), ("s", sp, np.array([0.0, -1.0]), 0.0, 0),
                       ("w", sp, np.array([-1.0, 0.0]), 0.0, 1), ("e", sp, np.array([1.0, 0.0]), 0.0, 0)]}
        approaches = {("cJ", "n", 1): ("cK", True), ("cJ", "s", 0): ("s1", True), ("cJ", "w", 1): ("cW", True),
                      ("cJ", "e", 0): ("cE", True)}
        import furniture
        orig = furniture.junction_controls
        furniture.junction_controls = lambda a, k, link_ok=None, ring=None, signals=None, country=None: [(x, furniture.G_STOP) for x in a]     # an all-way stop
        try:
            out = navgraph._signals(arms, approaches, nodes)
        finally:
            furniture.junction_controls = orig
        got = {ap[0] for ap in out[0]["app"]}
        self.assertEqual(got, {"s1", "cW", "cE"}, "the 8 m link from the next connector gets no stop")

    def test_signal_heads_face_their_drivers_in_left_hand_traffic(self):
        """The switchable boom keeps its right-hand spot and yaw where traffic drives on the left (furniture.prop_side:
        mirrored, it turned 180 degrees and faced away from its drivers, Basildon 2026-11-01), while the stop point
        moves to the left-hand lane."""
        import roads
        import furniture
        sp = roads.Spec(8.0, "paved", True, "residential", False)
        nodes = {"cJ": [0.0, 0.0, 0.0, 4], "cN": [0.0, 40.0, 0.0, 2], "cS": [0.0, -40.0, 0.0, 2],
                 "cW": [-40.0, 0.0, 0.0, 2], "cE": [40.0, 0.0, 0.0, 2]}
        arms = {"cJ": [("n", sp, np.array([0.0, 1.0]), 0.0, 1), ("s", sp, np.array([0.0, -1.0]), 0.0, 0),
                       ("w", sp, np.array([-1.0, 0.0]), 0.0, 1), ("e", sp, np.array([1.0, 0.0]), 0.0, 0)]}
        approaches = {("cJ", "n", 1): ("cN", True), ("cJ", "s", 0): ("cS", True), ("cJ", "w", 1): ("cW", True),
                      ("cJ", "e", 0): ("cE", True)}
        orig = furniture.junction_controls
        furniture.junction_controls = lambda a, k, link_ok=None, ring=None, signals=None, country=None: \
            [(x, furniture.G_SIGNAL) for x in a]
        try:
            rhd, lhd = (navgraph._signals(arms, approaches, nodes, left_hand=lh)[0]["app"] for lh in (False, True))
        finally:
            furniture.junction_controls = orig
        self.assertEqual(len(rhd), 4)
        for r, l in zip(sorted(rhd), sorted(lhd)):
            self.assertIsNotNone(r[7])
            self.assertEqual(r[7:], l[7:], "the head stands and faces the same way whichever side traffic drives on")
            inc = np.array(r[4:6])
            stop_r, stop_l = np.array(r[1:3]), np.array(l[1:3])
            left = np.array([-inc[1], inc[0]])
            self.assertLess(float(stop_r @ left), 0.0, "right-hand traffic stops in the right-hand lane")
            self.assertGreater(float(stop_l @ left), 0.0, "left-hand traffic stops in the left-hand lane")

    def test_left_hand_signal_falls_back_to_the_nearside_kerb_facing_the_same_way(self):
        """A dual carriageway's offside is its central reservation: no clear spot there, the head stands on the nearside
        kerb, still turned to face the drivers (5 of 13 heads of a Basildon junction were dropped, 2026-11-01)."""
        import furniture
        u = np.array([1.0, 0.0])                            # traffic comes in heading west; its offside is north
        other = [(4.0, np.array([[-50.0, 9.0], [80.0, 9.0]]))]     # the opposite carriageway, north of it
        own = [(4.0, np.array([[0.0, 0.0], [80.0, 0.0]]))]
        q, yaw = furniture.junction_spot(np.zeros(2), u, furniture.G_SIGNAL, 4.0, 4.0, own + other, left_hand=True)
        self.assertLess(q[1], 0.0, "on the nearside (south) kerb")
        self.assertAlmostEqual(yaw, furniture._yaw(0.0, 1.0), msg="turned as on the offside: facing the drivers")
        q2, _ = furniture.junction_spot(np.zeros(2), u, furniture.G_SIGNAL, 4.0, 4.0, own, left_hand=True)
        self.assertGreater(q2[1], 0.0, "the offside when it is clear")
        self.assertIsNone(furniture.junction_spot(np.zeros(2), u, furniture.G_SIGNAL, 4.0, 4.0, own + other),
                          "right-hand traffic: unchanged (no fallback)")

    def test_prop_side(self):
        import furniture
        u = np.array([1.0, 0.0])            # an arm going east: its traffic comes in heading west
        north, south = np.array([0.0, 1.0]), np.array([0.0, -1.0])
        self.assertTrue(np.allclose(furniture.prop_side(u, furniture.G_STOP), north), "a sign on the driver's right")
        self.assertTrue(np.allclose(furniture.prop_side(u, furniture.G_YIELD, True), south), "... on the left in GB")
        for lh in (False, True):
            self.assertTrue(np.allclose(furniture.prop_side(u, furniture.G_SIGNAL, lh), north))


class RulesTest(unittest.TestCase):
    def test_default_speeds_by_country(self):
        class F:            # a feature without a mapped limit
            rules = {}
        mw = roads.Spec(11.0, "paved", False, "motorway", True)
        self.assertAlmostEqual(navgraph.road_mph(F, 0, mw), 65.0)
        self.assertAlmostEqual(navgraph.road_mph(F, 0, mw, mph=False) * 1.609344, 120.0)
        town = roads.Spec(9.0, "paved", True, "primary", False)
        self.assertAlmostEqual(navgraph.road_mph(F, 0, town, mph=False), 35.0, msg="in town: the urban cap")
        res = roads.Spec(6.0, "paved", True, "residential", False)
        self.assertAlmostEqual(navgraph.road_mph(F, 0, res, mph=False), 25.0)

    def test_great_britain_national_limits_where_none_is_mapped(self):
        """Unmapped roads in GB take the national limits (30 mph restricted roads in town, 60 single and 70 dual
        carriageways, 70 motorways), never the US class defaults' 25 and 35 (Basildon, 2026-11-02)."""
        class F:
            rules = {}
        S = lambda cls, urban, oneway=False, median=None: roads.Spec(7.0, "paved", urban, cls, oneway,  # noqa: E731
                                                                       median=median)
        cases = [(S("residential", True), 30), (S("tertiary", True), 30), (S("primary", True), 30),
                 (S("trunk", True, True, 12.0), 70), (S("motorway", True, True, 12.0), 70),
                 (S("residential", False), 60), (S("primary", False), 60), (S("primary", False, True, 12.0), 70),
                 (S("motorway", False, True), 70), (S("trunk", False), 60)]
        for sp, want in cases:
            with self.subTest(cls=sp.cls, urban=sp.urban, oneway=sp.oneway):
                self.assertEqual(navgraph.road_mph(F, 0, sp, country="GB"), want)
                self.assertEqual(navgraph.road_mph(F, 0, sp, country="gb"), want)
        self.assertEqual(navgraph.road_mph(F, 0, S("service", True), country="GB"), navgraph.DRIVE["service"][1])
        self.assertEqual(navgraph.road_mph(F, 0, S("residential", True)), 25.0, "elsewhere: unchanged")
        self.assertEqual(navgraph.road_mph(F, 0, S("tertiary", True), country="US"), 30.0)

    def test_an_unmapped_piece_takes_the_limit_of_the_street_it_joins(self):
        """OpenStreetMap maps a street's limit on all but a piece (Southernhay, Basildon): the piece takes the value
        of the same-named pieces it joins when they agree; otherwise the default stands."""
        lines = [[(0, 0), (100, 0)], [(100, 0), (140, 0)], [(140, 0), (300, 0)], [(140, 0), (140, 200)]]
        f = features(lines, ["tertiary"] * 4)
        f.attrs["name"] = np.array(["Southernhay", "Southernhay", "Southernhay", "Other Road"])
        mph30 = 30 * 1.609344
        f.rules["max_speed_kmh"] = rules([(0, mph30, 0, 1), (2, mph30, 0, 1), (3, 64.4, 0, 1)], "float")
        sp = roads.Spec(7.0, "paved", True, "tertiary", False)
        res = roads.Spec(7.0, "paved", True, "residential", False)
        self.assertAlmostEqual(navgraph.road_mph(f, 1, res, country="US"), 30.0, delta=0.05,
                               msg="the neighbours' 30, not the residential default's 25")
        self.assertAlmostEqual(navgraph.road_mph(f, 1, sp), 30.0, delta=0.05)
        # the neighbours disagree: no inheritance, the default
        f2 = features(lines, ["tertiary"] * 4)
        f2.attrs["name"] = f.attrs["name"]
        f2.rules["max_speed_kmh"] = rules([(0, mph30, 0, 1), (2, 64.4, 0, 1)], "float")
        self.assertEqual(navgraph.inherited_kmh(f2, 1), None)
        self.assertAlmostEqual(navgraph.road_mph(f2, 1, sp, country="GB"), 30.0)
        # one mapped neighbour of 40 mph on one end only: taken
        f3 = features(lines, ["tertiary"] * 4)
        f3.attrs["name"] = f.attrs["name"]
        f3.rules["max_speed_kmh"] = rules([(2, 40 * 1.609344, 0, 1)], "float")
        self.assertAlmostEqual(navgraph.road_mph(f3, 1, sp, country="GB"), 40.0, delta=0.05)
        # another street's limit at the shared junction is not taken
        f4 = features(lines, ["tertiary"] * 4)
        f4.attrs["name"] = f.attrs["name"]
        f4.rules["max_speed_kmh"] = rules([(3, 64.4, 0, 1)], "float")
        self.assertIsNone(navgraph.inherited_kmh(f4, 1))

    def test_driving_side_and_signals_by_country(self):
        self.assertEqual(navgraph.road_rules("US"), {"rightHandDrive": False, "turnOnRed": True, "redYellow": False,
                                                     "country": "US", "mph": True})
        self.assertFalse(navgraph.road_rules("CH")["mph"])
        self.assertTrue(navgraph.road_rules("GB")["mph"])
        self.assertTrue(navgraph.road_rules("gb")["rightHandDrive"], "the game's name: true = driving on the left")
        self.assertTrue(navgraph.road_rules("GB")["redYellow"])
        self.assertFalse(navgraph.road_rules("DE")["rightHandDrive"])
        self.assertTrue(navgraph.road_rules("JP")["rightHandDrive"])
        self.assertFalse(navgraph.road_rules(None)["rightHandDrive"], "unknown: US rules")


if __name__ == "__main__":
    unittest.main()
