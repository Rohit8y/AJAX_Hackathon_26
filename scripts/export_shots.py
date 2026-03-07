"""
Export all shot clips from shots_trimmed.parquet to JSON for the 3D visualizer.
Also writes shots_index.json with metadata for the frontend.
"""

import json
import os
import numpy as np
import pandas as pd

# Body part ID (1-21) -> name mapping (from shots_parts.parquet)
PART_NAMES = {
    1: "LEFT_EAR",
    2: "NOSE",
    3: "RIGHT_EAR",
    4: "LEFT_SHOULDER",
    5: "NECK",
    6: "RIGHT_SHOULDER",
    7: "LEFT_ELBOW",
    8: "RIGHT_ELBOW",
    9: "LEFT_WRIST",
    10: "RIGHT_WRIST",
    11: "LEFT_HIP",
    12: "PELVIS",
    13: "RIGHT_HIP",
    14: "LEFT_KNEE",
    15: "RIGHT_KNEE",
    16: "LEFT_ANKLE",
    17: "RIGHT_ANKLE",
    18: "LEFT_HEEL",
    19: "LEFT_TOE",
    20: "RIGHT_HEEL",
    21: "RIGHT_TOE",
}

# Joint names in a stable order (index 0..20)
JOINT_NAMES = [PART_NAMES[i] for i in range(1, 22)]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "viz", "public", "data")


def main():
    print("Loading shots_trimmed.parquet...")
    df = pd.read_parquet(
        os.path.join(os.path.dirname(__file__), "..", "data", "shots_trimmed.parquet")
    )

    # Sort by frame number
    df = df.sort_values("frame_number").reset_index(drop=True)
    frames_arr = df["frame_number"].values

    # Find clip boundaries: gaps > 500 frames indicate new shot
    diffs = np.diff(frames_arr)
    gap_indices = np.where(diffs > 500)[0]

    # Build clip ranges: list of (start_idx, end_idx) in df
    clip_ranges = []
    start = 0
    for g in gap_indices:
        clip_ranges.append((start, g + 1))  # end is exclusive row index
        start = g + 1
    clip_ranges.append((start, len(df)))

    print(f"Found {len(clip_ranges)} clips total, exporting all.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    index = []

    for shot_idx in range(len(clip_ranges)):
        r_start, r_end = clip_ranges[shot_idx]
        clip_df = df.iloc[r_start:r_end]

        print(
            f"  Shot {shot_idx}: {len(clip_df)} frames "
            f"({clip_df['frame_number'].iloc[0]} – {clip_df['frame_number'].iloc[-1]})"
        )

        frames_out = []
        for _, row in clip_df.iterrows():
            # Ball position
            ball = row["ball"]
            if ball and row["ball_exists"]:
                ball_pos = [
                    round(float(ball["position_x"]), 3),
                    round(float(ball["position_y"]), 3),
                    round(float(ball["position_z"]), 3),
                ]
            else:
                ball_pos = None

            # Players
            players = []
            skeletons = row["skeletons"]
            if skeletons is not None:
                for skel in skeletons:
                    jersey = int(skel["jersey_number"])
                    team = int(skel["team"])

                    # Build a name->pos dict for this skeleton's parts
                    part_lookup = {}
                    for part in skel["parts"]:
                        name = PART_NAMES.get(int(part["name"]))
                        if name:
                            part_lookup[name] = [
                                round(float(part["position_x"]), 3),
                                round(float(part["position_y"]), 3),
                                round(float(part["position_z"]), 3),
                            ]

                    # Export joints in stable order (None if missing)
                    pos = [part_lookup.get(jn) for jn in JOINT_NAMES]

                    players.append({"jersey": jersey, "team": team, "pos": pos})

            frames_out.append(
                {
                    "f": int(row["frame_number"]),
                    "ball": ball_pos,
                    "players": players,
                }
            )

        shot_data = {
            "fps": 25,
            "joint_names": JOINT_NAMES,
            "frames": frames_out,
        }

        out_path = os.path.join(OUTPUT_DIR, f"shot_{shot_idx}.json")
        with open(out_path, "w") as f:
            json.dump(shot_data, f, separators=(",", ":"))  # compact

        size_kb = os.path.getsize(out_path) / 1024
        print(f"  -> {out_path} ({size_kb:.0f} KB)")

        index.append({
            "idx": shot_idx,
            "frames": len(frames_out),
            "duration": round(len(frames_out) / 25, 1),
            "first_frame": int(clip_df["frame_number"].iloc[0]),
            "last_frame": int(clip_df["frame_number"].iloc[-1]),
        })

    index_path = os.path.join(OUTPUT_DIR, "shots_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, separators=(",", ":"))
    print(f"Index -> {index_path}")
    print("Done.")


if __name__ == "__main__":
    main()
