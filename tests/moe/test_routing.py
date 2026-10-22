import unittest
from routing import make_routes, pack_routes


class RoutingTest(unittest.TestCase):
    def test_unique_slots_and_conservation(self):
        for pattern in ["balanced", "random", "hotspot"]:
            routes, slots, counts = make_routes(17, 64, 8, pattern, 42)
            self.assertEqual(sum(counts), 17 * 8)
            self.assertEqual(len(set(slots)), 17 * 8)
            for row in routes:
                self.assertEqual(len(set(row)), 8)
            self.assertTrue(all(0 <= x < 17 * 64 for x in slots))

    def test_balance_and_skew(self):
        self.assertEqual(make_routes(16)[2], [2] * 64)
        self.assertEqual(make_routes(16, pattern="hotspot")[2], [16] * 8 + [0] * 56)
        self.assertEqual(make_routes(17, pattern="random"), make_routes(17, pattern="random"))

    def test_invalid_routes(self):
        for rows in [[[0, 0]], [[64]], [[]], [[0, 1], [2]]]:
            with self.assertRaises(ValueError):
                pack_routes(rows, 64)


if __name__ == "__main__":
    unittest.main()
