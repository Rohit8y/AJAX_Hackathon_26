"""
Sample script: extract skeleton data for a specific event category.
Run with: .venv/bin/python3 sample_event_slice.py
"""

from skeleton_data import SkeletonData, BodyPart, Team
from event_data import EventParser

PARQUET = "/Users/rohityadav/Documents/AJAX/HackathonData/anonymized-limbtracking.parquet"
XML     = "data/XML_anonymized.xml"

# ── Load data ─────────────────────────────────────────────────────────────────

data   = SkeletonData(PARQUET)
parser = EventParser(XML, data)

# ── Choose your event category ────────────────────────────────────────────────
# Uncomment any of these, or combine multiple into a list:

EVENT_CODES = [
    #"AJAX | SCHOT",          # Ajax shots
    "AJAX | DOELPUNT",       # Ajax goals
    # "FORTUNA SITTARD | SCHOT",     # Opponent shots
    # "FORTUNA SITTARD | DOELPUNT",  # Opponent goals
    # "AJAX | KANS",                 # Ajax chances
    # "AJAX | CORNER",               # Ajax corners
]

# Optional: extend the window around each event (seconds)
PAD_BEFORE = 3.0   # seconds before event start
PAD_AFTER  = 2.0   # seconds after event end

# ── Option A: structured Frame objects ────────────────────────────────────────
# Best when you want to write custom per-frame logic.

print("=" * 60)
print("OPTION A — Structured Frame objects per event")
print("=" * 60)

results = parser.get_frames_for_events(EVENT_CODES, PAD_BEFORE, PAD_AFTER)

for event, frames in results.items():
    print(f"\n{event}")
    print(f"  Tagged players : {event.players_tagged}")
    print(f"  Frames loaded  : {len(frames)}")

    if not frames:
        continue

    first = frames[0]
    if first.ball:
        print(f"  Ball speed (frame 0): {first.ball.speed:.2f} m/s")

    # Example: get pelvis position of each player in first frame
    for player in first.players[:3]:
        pelvis = player.get_position(BodyPart.PELVIS)
        if pelvis:
            print(f"  Player {player.jersey_number:>2} ({player.team.name if player.team else '?'}): "
                  f"pelvis=({pelvis.x:.1f}, {pelvis.y:.1f}) cm")

# ── Option B: flat DataFrame (best for analysis) ──────────────────────────────
# One row per (frame × player × body_part).

print("\n" + "=" * 60)
print("OPTION B — Flat DataFrame (frame × player × body_part)")
print("=" * 60)

df = parser.get_flat_df_for_events(EVENT_CODES, PAD_BEFORE, PAD_AFTER)
print(f"\nShape: {df.shape}")
print(f"Events included: {df['event_id'].nunique()}")
print(f"Unique players : {df['jersey_number'].nunique()}")
print(f"Columns        : {list(df.columns)}")
print()
print(df[["event_code", "frame_number", "jersey_number", "team_name",
          "body_part_name", "pos_x", "pos_y", "pos_z"]].head(8).to_string(index=False))

# Example aggregation: average positions per body part per event type
print("\n--- Average pos_x / pos_y per body part (across all matching events) ---")
agg = (
    df.groupby(["event_code", "body_part_name"])[["pos_x", "pos_y"]]
    .mean()
    .round(1)
)
print(agg.head(12).to_string())

# ── Option C: player-level DataFrame ─────────────────────────────────────────
# One row per (frame × player), position = Pelvis.

print("\n" + "=" * 60)
print("OPTION C — Player-level DataFrame (Pelvis position per frame)")
print("=" * 60)

pf = parser.get_player_df_for_events(EVENT_CODES, PAD_BEFORE, PAD_AFTER)
print(f"\nShape: {pf.shape}")
print()
print(pf[["event_id", "event_code", "frame_number", "jersey_number",
          "team_name", "pos_x", "pos_y", "ball_x", "ball_y"]].head(8).to_string(index=False))

# Example: how far is each player from the ball on average during these events?
import numpy as np
pf["dist_to_ball_m"] = np.sqrt(
    (pf["pos_x"] - pf["ball_x"]) ** 2 +
    (pf["pos_y"] - pf["ball_y"]) ** 2
) / 100  # cm → m

print("\n--- Average distance to ball per team during these events ---")
print(
    pf.groupby("team_name")["dist_to_ball_m"]
    .mean()
    .round(2)
    .to_string()
)
