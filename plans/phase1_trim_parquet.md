# Shot Frame Detection — Phase 1: Trim Parquet to Shot Windows

## Goal
Reduce `anonymized-limbtracking.parquet` (167K frames) to only frames near shot/goal events.
Output: `data/shots_trimmed.parquet` — same schema, smaller dataset for Phase 2 spike detection.

---

## Critical Corrections to Original Plan

| Issue | Fix |
|---|---|
| Uses `DOELPUNT` only | Use `SCHOT` + `DOELPUNT` — 18 shots vs 7 goals |
| Uses `instances_parsed.csv` for timestamps | Use `EventParser` (XML) — phase calibration already done |
| 15 events out of parquet range | Skip gracefully, don't crash |

---

## New File: `build_shot_parquet.py`

**Reuses:** `SkeletonData` (`skeleton_data.py:301`), `EventParser.get_events()` (`event_data.py:200`)

```
Constants (top of file, all configurable):
  PARQUET_IN  = "~/Downloads/HackathonData/anonymized-limbtracking.parquet"
  XML         = "data/XML_anonymized.xml"
  PARQUET_OUT = "data/shots_trimmed.parquet"
  PAD_BEFORE  = 10.0   # seconds  ← placeholder
  PAD_AFTER   = 10.0   # seconds  ← placeholder
  EVENT_CODES = ["AJAX | SCHOT", "FORTUNA SITTARD | SCHOT",
                 "AJAX | DOELPUNT", "FORTUNA SITTARD | DOELPUNT"]

Steps:
  1. EventParser.get_events(EVENT_CODES, PAD_BEFORE, PAD_AFTER)
     → Event objects with .start_frame / .end_frame already converted
  2. Filter out events outside parquet frame range (skip silently)
  3. Sort + merge overlapping (start_frame, end_frame) windows
  4. Boolean mask over frame_number column → pq.write_table() to PARQUET_OUT
     (PyArrow preserves schema metadata automatically — framerate, phase info intact)
  5. Print: N events in-range, N windows after merge, N frames kept, % reduction
```

---

## Verification

```bash
.venv/bin/python build_shot_parquet.py
# Expected: ~10 in-range events, ~8 windows, ~4500 frames (~2.7% of original)

.venv/bin/python -c "
from skeleton_data import SkeletonData
d = SkeletonData('data/shots_trimmed.parquet')
print(d.metadata.framerate)  # 25.0
print(d.frame_count)         # ~4500
"
```

---

## Next Phase
**Phase 2**: Ball acceleration spike detection within each window → exact shot frame + shooter ID.
