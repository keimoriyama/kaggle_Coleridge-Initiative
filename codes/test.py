import unittest
from utils import get_time

class time_test(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_normal(self):
        start = 0
        end = 39
        expect = "39sec"
        sentence = get_time(start, end)
        self.assertEqual(expect, sentence)

    def test_min(self):
        start = 0
        end = 66
        expect = "1min 6sec"
        sentence = get_time(start, end)
        self.assertEqual(expect, sentence)

    def test_hour(self):
        start = 0
        end = 3700
        expect = "1h 1min 40sec"
        sentence = get_time(start, end)
        self.assertEqual(expect, sentence)

if __name__ == "__main__":
    unittest.main()