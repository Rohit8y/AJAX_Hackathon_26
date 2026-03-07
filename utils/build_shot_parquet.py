"""
Phase 1: Trim parquet to frames around shot/goal events only.
Output: data/shots_trimmed.parquet (same schema + metadata, smaller file)
"""

import os
import numpy as np
import pyarrow.parquet as pq

from utils.skeleton_data import SkeletonData
from event_data import EventParser

# ── Config ────────────────────────────────────────────────────────────────────

PARQUET_IN  = os.path.expanduser("~/Downloads/HackathonData/anonymized-limbtracking.parquet")
XML         = "data/XML_anonymized.xml"
PARQUET_OUT = "data/shots_trimmed.parquet"

PAD_BEFORE  = 10.0   # seconds before event start  ← placeholder, tune as needed
PAD_AFTER   = 10.0   # seconds after event end      ← placeholder, tune as needed

EVENT_CODES = [
    "AJAX | SCHOT",
    "FORTUNA SITTARD | SCHOT",
    "AJAX | DOELPUNT",
    "FORTUNA SITTARD | DOELPUNT",
]

# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading parquet and XML...")
data   = SkeletonData(PARQUET_IN)
parser = EventParser(XML, data)

# Full frame range of this parquet file
data._load()
frame_col  = np.array(data._table.column("frame_number").to_pylist(), dtype=np.int64)
parquet_min, parquet_max = int(frame_col.min()), int(frame_col.max())
print(f"Parquet frame range : {parquet_min} – {parquet_max}  ({len(frame_col):,} frames)")

# ── Collect event windows ─────────────────────────────────────────────────────

events = parser.get_events(EVENT_CODES, PAD_BEFORE, PAD_AFTER)
print(f"Events found in XML : {len(events)}")

in_range, skipped = [], 0
for e in events:
    if e.end_frame < parquet_min or e.start_frame > parquet_max:
        skipped += 1
        continue
    # Clamp to parquet bounds
    start = max(e.start_frame, parquet_min)
    end   = min(e.end_frame,   parquet_max)
    in_range.append((start, end))

print(f"In-range events     : {len(in_range)}  ({skipped} skipped — outside this parquet)")

# ── Merge overlapping windows ─────────────────────────────────────────────────

in_range.sort()
merged = []
for start, end in in_range:
    if merged and start <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], end)
    else:
        merged.append([start, end])

print(f"Windows after merge : {len(merged)}")

# ── Build mask and filter ─────────────────────────────────────────────────────

mask = np.zeros(len(frame_col), dtype=bool)
for start, end in merged:
    mask |= (frame_col >= start) & (frame_col <= end)

indices = np.where(mask)[0]
kept    = len(indices)
print(f"Frames kept         : {kept:,} / {len(frame_col):,}  ({100*kept/len(frame_col):.1f}%)")

# ── Write ─────────────────────────────────────────────────────────────────────

filtered_table = data._table.take(indices)
os.makedirs(os.path.dirname(PARQUET_OUT) or ".", exist_ok=True)
pq.write_table(filtered_table, PARQUET_OUT)
print(f"Written to          : {PARQUET_OUT}")

# ── Sanity check ──────────────────────────────────────────────────────────────

check = SkeletonData(PARQUET_OUT)
print(f"\nSanity check:")
print(f"  framerate  : {check.metadata.framerate}")
print(f"  frame count: {check.frame_count:,}")
print("Done.")
