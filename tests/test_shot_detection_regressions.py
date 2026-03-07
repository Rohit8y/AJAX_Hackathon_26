import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "shot_detections.csv"
SCRIPT_PATH = ROOT / "utils" / "detect_shot_frames.py"


class ShotDetectionRegressionTests(unittest.TestCase):
    def test_goal_frames_remain_stable(self) -> None:
        subprocess.run([sys.executable, str(SCRIPT_PATH)], check=True, cwd=ROOT)

        out = pd.read_csv(CSV_PATH).set_index("event_id")
        expected_goal_frames = {
            45: 1810682,
            167: 1821017,
            318: 1836288,
            640: 1868118,
            1226: 1951882,
        }

        for event_id, expected_frame in expected_goal_frames.items():
            with self.subTest(event_id=event_id):
                self.assertEqual(int(out.loc[event_id, "shot_frame"]), expected_frame)


if __name__ == "__main__":
    unittest.main()
