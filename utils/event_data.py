"""
Event data parser for the XML event file (Sportscode/SportsCode format).
Combines with SkeletonData to extract parquet frames for specific event types.

Event codes in this match:
  Shots:  "AJAX | SCHOT", "FORTUNA SITTARD | SCHOT"
  Goals:  "AJAX | DOELPUNT", "FORTUNA SITTARD | DOELPUNT"
  Chances:"AJAX | KANS", "FORTUNA SITTARD | KANS"
  Corners:"AJAX | CORNER", "FORTUNA SITTARD | CORNER"
  Freekicks: "AJAX | VRIJE TRAP OVERIG", "FORTUNA SITTARD | VRIJE TRAP OVERIG"
  Goal kicks: "AJAX | DOELTRAP", "FORTUNA SITTARD | DOELTRAP"
  Throw-ins: "AJAX | INGOOI", "FORTUNA SITTARD | INGOOI"
  Phases: "AANVALLEN", "VERDEDIGEN", "BOP", "V>A", "A>V"
  Summary: "SUMMARY"

Time mapping:
  XML times are in seconds from video start.
  Parquet frame_number uses a fixed offset: frame = (xml_time + offset) * framerate
  The offset is calibrated automatically from the START events in the XML
  and the phase start frames from the parquet metadata.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from skeleton_data import SkeletonData, Phase, BodyPart, Frame, GameMetadata


# ── Event dataclass ───────────────────────────────────────────────────────────

@dataclass(eq=False)
class Event:
    """A single tagged event from the XML file."""
    id: int
    code: str
    start_sec: float       # seconds in XML video timeline
    end_sec: float         # seconds in XML video timeline
    start_frame: int       # corresponding parquet frame number
    end_frame: int         # corresponding parquet frame number
    labels: dict[str, list[str]] = field(default_factory=dict)
    # labels keys are group names (e.g. "SPELERS", "HOOFDMOMENTEN", "FASE")
    # labels values are lists of texts within that group

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def players_tagged(self) -> list[str]:
        """Player names tagged in this event (from SPELERS label group)."""
        return self.labels.get("SPELERS", [])

    @property
    def phase_tag(self) -> Optional[str]:
        """The FASE label if present."""
        phases = self.labels.get("FASE", [])
        return phases[0] if phases else None

    def __repr__(self) -> str:
        return (
            f"Event(id={self.id}, code={self.code!r}, "
            f"{self.start_sec:.1f}-{self.end_sec:.1f}s, "
            f"frames={self.start_frame}-{self.end_frame})"
        )


# ── Time calibration ──────────────────────────────────────────────────────────

@dataclass
class TimeCalibration:
    """
    Maps XML video-timeline seconds to parquet frame numbers.
    Derived from START events in the XML and phase start frames in parquet metadata.
    """
    phase1_offset: float   # seconds to add to xml_time before multiplying by framerate
    phase2_offset: float
    phase1_xml_boundary: float  # xml time of 2nd START event (start of 2nd half)
    framerate: float

    def xml_time_to_frame(self, xml_time: float) -> int:
        offset = self.phase1_offset if xml_time < self.phase1_xml_boundary else self.phase2_offset
        return round((xml_time + offset) * self.framerate)

    @classmethod
    def from_xml_and_meta(
        cls,
        xml_start_events: list[float],
        meta: GameMetadata,
    ) -> "TimeCalibration":
        """
        xml_start_events: sorted list of xml times for START-coded events.
        meta: GameMetadata from the parquet file.
        """
        if len(xml_start_events) < 2:
            raise ValueError(
                "Need at least 2 START events in XML to calibrate phase 1 and phase 2 offsets."
            )
        p1_info = meta.frames_for_phase(Phase.FIRST_HALF)
        p2_info = meta.frames_for_phase(Phase.SECOND_HALF)
        if p1_info is None or p2_info is None:
            raise ValueError("Parquet metadata missing phase 1 or phase 2 frame ranges.")

        off1 = (p1_info.start_frame / meta.framerate) - xml_start_events[0]
        off2 = (p2_info.start_frame / meta.framerate) - xml_start_events[1]

        return cls(
            phase1_offset=off1,
            phase2_offset=off2,
            phase1_xml_boundary=xml_start_events[1],
            framerate=meta.framerate,
        )


# ── XML parser ────────────────────────────────────────────────────────────────

class EventParser:
    """
    Parse the Sportscode XML event file.

    Usage:
        parser = EventParser("data/XML_anonymized.xml", skeleton_data)
        events = parser.get_events(["AJAX | SCHOT", "AJAX | DOELPUNT"])
    """

    def __init__(self, xml_path: str, skeleton_data: SkeletonData):
        self.xml_path = xml_path
        self.skeleton_data = skeleton_data
        self._events: Optional[list[Event]] = None
        self._calibration: Optional[TimeCalibration] = None

    def _parse(self):
        if self._events is not None:
            return

        meta = self.skeleton_data.metadata
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        all_inst = root.find("ALL_INSTANCES")

        # Collect START event times for calibration
        start_times: list[float] = []
        for inst in all_inst.findall("instance"):
            code_el = inst.find("code")
            if code_el is not None and code_el.text == "START":
                start_times.append(float(inst.find("start").text))
        start_times.sort()

        self._calibration = TimeCalibration.from_xml_and_meta(start_times, meta)

        # Parse all instances
        events: list[Event] = []
        for inst in all_inst.findall("instance"):
            code_el = inst.find("code")
            if code_el is None:
                continue
            code = code_el.text or ""
            start_sec = float(inst.find("start").text)
            end_sec   = float(inst.find("end").text)

            # Parse label groups
            labels: dict[str, list[str]] = {}
            for label in inst.findall("label"):
                group = label.find("group")
                text  = label.find("text")
                if group is not None and text is not None:
                    g = group.text or ""
                    labels.setdefault(g, []).append(text.text or "")

            start_frame = self._calibration.xml_time_to_frame(start_sec)
            end_frame   = self._calibration.xml_time_to_frame(end_sec)

            events.append(Event(
                id=int(inst.find("ID").text),
                code=code,
                start_sec=start_sec,
                end_sec=end_sec,
                start_frame=start_frame,
                end_frame=end_frame,
                labels=labels,
            ))

        self._events = events

    @property
    def all_events(self) -> list[Event]:
        self._parse()
        return self._events

    @property
    def all_codes(self) -> list[str]:
        """Sorted list of all unique event codes."""
        self._parse()
        return sorted(set(e.code for e in self._events))

    def get_events(
        self,
        codes: list[str],
        pad_before_sec: float = 0.0,
        pad_after_sec: float = 0.0,
    ) -> list[Event]:
        """
        Return events matching any of the given codes.

        Args:
            codes: List of event codes to filter on, e.g. ["AJAX | SCHOT", "AJAX | DOELPUNT"]
            pad_before_sec: Extend each event's start backwards by this many seconds.
            pad_after_sec:  Extend each event's end forwards by this many seconds.
        """
        self._parse()
        code_set = set(codes)
        matched = [e for e in self._events if e.code in code_set]

        if pad_before_sec or pad_after_sec:
            fps = self._calibration.framerate
            pad_before_frames = round(pad_before_sec * fps)
            pad_after_frames  = round(pad_after_sec  * fps)
            matched = [
                Event(
                    id=e.id,
                    code=e.code,
                    start_sec=e.start_sec - pad_before_sec,
                    end_sec=e.end_sec + pad_after_sec,
                    start_frame=e.start_frame - pad_before_frames,
                    end_frame=e.end_frame + pad_after_frames,
                    labels=e.labels,
                )
                for e in matched
            ]

        return matched

    # ── Skeleton extraction ────────────────────────────────────────────────────

    def get_frames_for_events(
        self,
        codes: list[str],
        pad_before_sec: float = 0.0,
        pad_after_sec: float = 0.0,
    ) -> dict[Event, list[Frame]]:
        """
        Return a dict mapping each matched Event → list of skeleton Frame objects.
        Use this for structured per-event analysis.

        Example:
            results = parser.get_frames_for_events(["AJAX | SCHOT"])
            for event, frames in results.items():
                print(event, "→", len(frames), "frames")
                for f in frames:
                    print(f.ball.speed if f.ball else "no ball")
        """
        events = self.get_events(codes, pad_before_sec, pad_after_sec)
        result: dict[Event, list[Frame]] = {}
        for event in events:
            frames = self.skeleton_data.frames(event.start_frame, event.end_frame)
            result[event] = frames
        return result

    def get_flat_df_for_events(
        self,
        codes: list[str],
        pad_before_sec: float = 0.0,
        pad_after_sec: float = 0.0,
        include_ball: bool = True,
    ) -> pd.DataFrame:
        """
        Return a flat DataFrame (frame × player × body_part) for all matched events.
        Adds columns: event_id, event_code, event_start_frame, event_end_frame.

        Use this for aggregate analysis across many events.

        Example:
            df = parser.get_flat_df_for_events(["AJAX | SCHOT", "AJAX | DOELPUNT"])
            # average body part positions per event
            df.groupby(["event_code", "body_part_name"])[["pos_x","pos_y","pos_z"]].mean()
        """
        events = self.get_events(codes, pad_before_sec, pad_after_sec)
        dfs = []
        for event in events:
            df = self.skeleton_data.to_flat_dataframe(
                start_frame=event.start_frame,
                end_frame=event.end_frame,
                include_ball=include_ball,
            )
            if not df.empty:
                df["event_id"]          = event.id
                df["event_code"]        = event.code
                df["event_start_frame"] = event.start_frame
                df["event_end_frame"]   = event.end_frame
                dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def get_player_df_for_events(
        self,
        codes: list[str],
        pad_before_sec: float = 0.0,
        pad_after_sec: float = 0.0,
        key_part: BodyPart = BodyPart.PELVIS,
    ) -> pd.DataFrame:
        """
        Return a player-level DataFrame (frame × player) for all matched events.
        Uses key_part (default: PELVIS) as the player's position.
        Adds columns: event_id, event_code.

        Example:
            df = parser.get_player_df_for_events(["AJAX | SCHOT"], pad_before_sec=2)
            # see where players are 2 seconds before each shot
        """
        events = self.get_events(codes, pad_before_sec, pad_after_sec)
        dfs = []
        for event in events:
            df = self.skeleton_data.to_player_dataframe(
                start_frame=event.start_frame,
                end_frame=event.end_frame,
                key_part=key_part,
            )
            if not df.empty:
                df["event_id"]   = event.id
                df["event_code"] = event.code
                dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)
