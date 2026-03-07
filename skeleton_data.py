"""
Skeleton data classes and loader for TRACAB TF15 Parquet format.

Data is in cm from center of field (x, y) and cm above ground (z).
Ball velocity is in m/s. Framerate is typically 25fps.

Team values: 1=Home, 0=Away, 3=Referee
Body part IDs: see BodyPart enum
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import pyarrow.parquet as pq
import pandas as pd
import numpy as np


# ── Enums ─────────────────────────────────────────────────────────────────────

class BodyPart(IntEnum):
    LEFT_EAR       = 1
    NOSE           = 2
    RIGHT_EAR      = 3
    LEFT_SHOULDER  = 4
    NECK           = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW     = 7
    RIGHT_ELBOW    = 8
    LEFT_WRIST     = 9
    RIGHT_WRIST    = 10
    LEFT_HIP       = 11
    PELVIS         = 12
    RIGHT_HIP      = 13
    LEFT_KNEE      = 14
    RIGHT_KNEE     = 15
    LEFT_ANKLE     = 16
    RIGHT_ANKLE    = 17
    LEFT_HEEL      = 18
    LEFT_TOE       = 19
    RIGHT_HEEL     = 20
    RIGHT_TOE      = 21

    @classmethod
    def from_id(cls, part_id: int) -> Optional["BodyPart"]:
        try:
            return cls(part_id)
        except ValueError:
            return None


class Team(IntEnum):
    """
    Team identifiers per TF15 spec: 0=Away, 1=Home, 3=Referee.
    Note: actual data may use other encodings (e.g. 2, 3, 4) as the spec
    states "Other values are used for internal purposes."
    Check your specific file's observed values.
    """
    AWAY    = 0
    HOME    = 1
    REFEREE = 3
    # Observed values in anonymized dataset (non-standard):
    TEAM_A  = 2  # first team in anonymized data
    TEAM_B  = 4  # second team (appears to be referees/assistants)

    @classmethod
    def from_id(cls, team_id: int) -> Optional["Team"]:
        try:
            return cls(team_id)
        except ValueError:
            return None

    @property
    def is_playing_team(self) -> bool:
        """True for HOME, AWAY, or the anonymized TEAM_A/TEAM_B playing values."""
        return self in (Team.HOME, Team.AWAY, Team.TEAM_A)


class Phase(IntEnum):
    FIRST_HALF    = 1
    SECOND_HALF   = 2
    EXTRA_TIME_1  = 3
    EXTRA_TIME_2  = 4
    PENALTIES     = 5


# ── Simple value dataclasses ───────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Position3D:
    """Position in cm from field center. Z is height above ground."""
    x: float  # cm, positive = right
    y: float  # cm, positive = up pitch
    z: float  # cm, height

    def distance_2d(self, other: "Position3D") -> float:
        """2D Euclidean distance in cm."""
        return float(np.hypot(self.x - other.x, self.y - other.y))

    def distance_3d(self, other: "Position3D") -> float:
        """3D Euclidean distance in cm."""
        return float(np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2))


@dataclass(frozen=True, slots=True)
class Velocity3D:
    """Velocity in m/s."""
    vx: float
    vy: float
    vz: float

    @property
    def speed(self) -> float:
        """Total speed magnitude in m/s."""
        return float(np.sqrt(self.vx**2 + self.vy**2 + self.vz**2))

    @property
    def speed_2d(self) -> float:
        """Horizontal speed magnitude in m/s."""
        return float(np.hypot(self.vx, self.vy))


@dataclass(frozen=True, slots=True)
class SkeletonPart:
    body_part: BodyPart
    position: Position3D


@dataclass
class Player:
    jersey_number: int   # -1 = unassigned; 0=head referee, 1=1st asst, 2=2nd asst
    team: Optional[Team]
    parts: dict[BodyPart, Position3D]  # keyed by BodyPart

    def get_position(self, part: BodyPart) -> Optional[Position3D]:
        return self.parts.get(part)

    @property
    def pelvis(self) -> Optional[Position3D]:
        return self.parts.get(BodyPart.PELVIS)

    @property
    def is_referee(self) -> bool:
        return self.team == Team.REFEREE or self.team == Team.TEAM_B


@dataclass
class Ball:
    position: Position3D
    velocity: Velocity3D

    @property
    def speed(self) -> float:
        return self.velocity.speed


@dataclass
class Frame:
    frame_number: int
    players: list[Player]
    ball: Optional[Ball]   # None when ball_exists=False

    def get_player(self, jersey_number: int, team: Optional[Team] = None) -> Optional[Player]:
        for p in self.players:
            if p.jersey_number == jersey_number:
                if team is None or p.team == team:
                    return p
        return None

    @property
    def home_players(self) -> list[Player]:
        return [p for p in self.players if p.team == Team.HOME]

    @property
    def away_players(self) -> list[Player]:
        return [p for p in self.players if p.team == Team.AWAY]

    @property
    def referees(self) -> list[Player]:
        return [p for p in self.players if p.team in (Team.REFEREE, Team.TEAM_B)]

    def players_by_team(self, team: Team) -> list[Player]:
        return [p for p in self.players if p.team == team]


# ── Metadata ──────────────────────────────────────────────────────────────────

@dataclass
class PhaseInfo:
    phase: Phase
    start_frame: int
    end_frame: int


@dataclass
class GameMetadata:
    game_id: str
    vendor_id: str
    home_team_id: str
    away_team_id: str
    framerate: float
    data_quality: int        # 0=raw, 1=scrubbed
    ai_clicker: bool
    file_version: int
    pitch_long: int          # dm - long side of tracking area
    pitch_short: int         # dm - short side of tracking area
    pitch_padding_left: int  # dm
    pitch_padding_right: int # dm
    pitch_padding_top: int   # dm
    pitch_padding_bottom: int# dm
    phases: list[PhaseInfo]

    @classmethod
    def from_parquet_metadata(cls, meta: dict) -> "GameMetadata":
        def _int(k: str, default=0) -> int:
            return int(meta.get(k, default))

        def _float(k: str, default=0.0) -> float:
            return float(meta.get(k, default))

        phases = []
        for i in range(1, 6):
            start = _int(f"phase_{i}_start")
            end   = _int(f"phase_{i}_end")
            if start and end:
                phases.append(PhaseInfo(Phase(i), start, end))

        return cls(
            game_id=meta.get("game_id", ""),
            vendor_id=meta.get("vendor_id", ""),
            home_team_id=meta.get("home_team_id", ""),
            away_team_id=meta.get("away_team_id", ""),
            framerate=_float("framerate"),
            data_quality=_int("data_quality"),
            ai_clicker=bool(_int("ai_clicker")),
            file_version=_int("file_version"),
            pitch_long=_int("pitch_long"),
            pitch_short=_int("pitch_short"),
            pitch_padding_left=_int("pitch_padding_left"),
            pitch_padding_right=_int("pitch_padding_right"),
            pitch_padding_top=_int("pitch_padding_top"),
            pitch_padding_bottom=_int("pitch_padding_bottom"),
            phases=phases,
        )

    def frames_for_phase(self, phase: Phase) -> Optional[PhaseInfo]:
        for p in self.phases:
            if p.phase == phase:
                return p
        return None

    def seconds_to_frames(self, seconds: float, phase: Phase = Phase.FIRST_HALF) -> int:
        """Convert seconds since phase start to absolute frame number."""
        p = self.frames_for_phase(phase)
        if p is None:
            raise ValueError(f"Phase {phase} not found in metadata")
        return p.start_frame + int(seconds * self.framerate)

    def frames_to_seconds(self, frame_number: int, phase: Phase = Phase.FIRST_HALF) -> float:
        """Convert absolute frame number to seconds since phase start."""
        p = self.frames_for_phase(phase)
        if p is None:
            raise ValueError(f"Phase {phase} not found in metadata")
        return (frame_number - p.start_frame) / self.framerate


# ── Raw dict → dataclass parsers ──────────────────────────────────────────────

def _parse_player(raw: dict) -> Player:
    parts: dict[BodyPart, Position3D] = {}
    for p in raw.get("parts") or []:
        bp = BodyPart.from_id(p["name"])
        if bp is not None:
            parts[bp] = Position3D(p["position_x"], p["position_y"], p["position_z"])
    return Player(
        jersey_number=int(raw["jersey_number"]),
        team=Team.from_id(int(raw["team"])),
        parts=parts,
    )


def _parse_ball(raw: dict) -> Ball:
    return Ball(
        position=Position3D(raw["position_x"], raw["position_y"], raw["position_z"]),
        velocity=Velocity3D(raw["velocity_x"], raw["velocity_y"], raw["velocity_z"]),
    )


def _parse_frame(row: dict) -> Frame:
    players = [_parse_player(sk) for sk in (row["skeletons"] or [])]
    ball = _parse_ball(row["ball"]) if row.get("ball_exists") else None
    return Frame(frame_number=int(row["frame_number"]), players=players, ball=ball)


# ── Main loader ───────────────────────────────────────────────────────────────

class SkeletonData:
    """
    Load and slice TRACAB TF15 parquet skeleton data.

    Usage:
        data = SkeletonData("path/to/file.parquet")

        # Slice by frame range
        frames = data.frames(start_frame=1801359, end_frame=1810000)

        # Slice by phase
        frames = data.frames_for_phase(Phase.FIRST_HALF)

        # Slice by seconds into a phase
        frames = data.frames_by_seconds(0, 60, Phase.FIRST_HALF)  # first minute

        # Flat DataFrame for analysis (one row per player-part per frame)
        df = data.to_flat_dataframe(start_frame=1801359, end_frame=1810000)
    """

    def __init__(self, path: str):
        self.path = path
        self._table = None  # lazy load
        self._meta: Optional[GameMetadata] = None

    def _load(self):
        if self._table is None:
            self._table = pq.read_table(self.path)
            raw_meta = {
                k.decode(): v.decode()
                for k, v in self._table.schema.metadata.items()
            }
            self._meta = GameMetadata.from_parquet_metadata(raw_meta)

    @property
    def metadata(self) -> GameMetadata:
        self._load()
        return self._meta

    def _slice_table(self, start_frame: Optional[int], end_frame: Optional[int]):
        self._load()
        t = self._table
        frame_col = t.column("frame_number").to_pylist()
        frames_arr = np.array(frame_col)
        mask = np.ones(len(frames_arr), dtype=bool)
        if start_frame is not None:
            mask &= frames_arr >= start_frame
        if end_frame is not None:
            mask &= frames_arr <= end_frame
        indices = np.where(mask)[0]
        return t.take(indices)

    def frames(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> list[Frame]:
        """Return structured Frame objects for the given frame number range."""
        sliced = self._slice_table(start_frame, end_frame)
        rows = sliced.to_pydict()
        n = sliced.num_rows
        return [
            _parse_frame({col: rows[col][i] for col in rows})
            for i in range(n)
        ]

    def frames_for_phase(self, phase: Phase) -> list[Frame]:
        """Return all frames for a match phase."""
        p = self.metadata.frames_for_phase(phase)
        if p is None:
            return []
        return self.frames(p.start_frame, p.end_frame)

    def frames_by_seconds(
        self,
        start_sec: float,
        end_sec: float,
        phase: Phase = Phase.FIRST_HALF,
    ) -> list[Frame]:
        """
        Slice by seconds into a phase.
        E.g. frames_by_seconds(0, 60) → first minute of first half.
        """
        start_frame = self.metadata.seconds_to_frames(start_sec, phase)
        end_frame   = self.metadata.seconds_to_frames(end_sec,   phase)
        return self.frames(start_frame, end_frame)

    def to_flat_dataframe(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        phase: Optional[Phase] = None,
        include_ball: bool = True,
    ) -> pd.DataFrame:
        """
        Flatten nested skeleton data into a tidy DataFrame.
        One row per (frame × player × body_part).

        Columns:
            frame_number, jersey_number, team, team_name,
            body_part, body_part_name,
            pos_x, pos_y, pos_z

        Plus optional ball columns (ball_x, ball_y, ball_z, ball_vx, ball_vy, ball_vz)
        if include_ball=True.
        """
        if phase is not None:
            p = self.metadata.frames_for_phase(phase)
            start_frame = p.start_frame if p else start_frame
            end_frame   = p.end_frame   if p else end_frame

        sliced = self._slice_table(start_frame, end_frame)
        rows_dict = sliced.to_pydict()
        n = sliced.num_rows

        records = []
        for i in range(n):
            fn = rows_dict["frame_number"][i]
            ball_exists = rows_dict["ball_exists"][i]
            ball_raw = rows_dict["ball"][i] if ball_exists else None

            for sk in (rows_dict["skeletons"][i] or []):
                jersey = int(sk["jersey_number"])
                team_id = int(sk["team"])
                team_enum = Team.from_id(team_id)
                team_name = team_enum.name if team_enum else f"UNKNOWN({team_id})"

                for part in (sk.get("parts") or []):
                    bp = BodyPart.from_id(int(part["name"]))
                    rec = {
                        "frame_number":   fn,
                        "jersey_number":  jersey,
                        "team":           team_id,
                        "team_name":      team_name,
                        "body_part":      int(part["name"]),
                        "body_part_name": bp.name if bp else f"UNKNOWN({part['name']})",
                        "pos_x":          part["position_x"],
                        "pos_y":          part["position_y"],
                        "pos_z":          part["position_z"],
                    }
                    if include_ball:
                        if ball_raw:
                            rec["ball_x"]  = ball_raw["position_x"]
                            rec["ball_y"]  = ball_raw["position_y"]
                            rec["ball_z"]  = ball_raw["position_z"]
                            rec["ball_vx"] = ball_raw["velocity_x"]
                            rec["ball_vy"] = ball_raw["velocity_y"]
                            rec["ball_vz"] = ball_raw["velocity_z"]
                        else:
                            rec["ball_x"] = rec["ball_y"] = rec["ball_z"] = np.nan
                            rec["ball_vx"] = rec["ball_vy"] = rec["ball_vz"] = np.nan
                    records.append(rec)

        df = pd.DataFrame(records)
        if not df.empty:
            df["frame_number"]  = df["frame_number"].astype("int32")
            df["jersey_number"] = df["jersey_number"].astype("int8")
            df["team"]          = df["team"].astype("int8")
            df["body_part"]     = df["body_part"].astype("int8")
        return df

    def to_player_dataframe(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        phase: Optional[Phase] = None,
        key_part: BodyPart = BodyPart.PELVIS,
    ) -> pd.DataFrame:
        """
        One row per (frame × player), using a single reference body part (default: Pelvis)
        as the player's position. Useful for tracking player movement over time.

        Columns: frame_number, jersey_number, team, team_name,
                 pos_x, pos_y, pos_z, ball_x, ball_y, ball_z, ball_vx, ball_vy, ball_vz
        """
        if phase is not None:
            p = self.metadata.frames_for_phase(phase)
            start_frame = p.start_frame if p else start_frame
            end_frame   = p.end_frame   if p else end_frame

        sliced = self._slice_table(start_frame, end_frame)
        rows_dict = sliced.to_pydict()
        n = sliced.num_rows

        records = []
        for i in range(n):
            fn = rows_dict["frame_number"][i]
            ball_exists = rows_dict["ball_exists"][i]
            ball_raw = rows_dict["ball"][i] if ball_exists else None

            ball_rec = {}
            if ball_raw:
                ball_rec = {
                    "ball_x": ball_raw["position_x"],
                    "ball_y": ball_raw["position_y"],
                    "ball_z": ball_raw["position_z"],
                    "ball_vx": ball_raw["velocity_x"],
                    "ball_vy": ball_raw["velocity_y"],
                    "ball_vz": ball_raw["velocity_z"],
                }
            else:
                ball_rec = {k: np.nan for k in ["ball_x","ball_y","ball_z","ball_vx","ball_vy","ball_vz"]}

            for sk in (rows_dict["skeletons"][i] or []):
                jersey  = int(sk["jersey_number"])
                team_id = int(sk["team"])
                team_enum = Team.from_id(team_id)
                team_name = team_enum.name if team_enum else f"UNKNOWN({team_id})"

                # Find the key body part
                ref_part = next(
                    (p for p in (sk.get("parts") or []) if int(p["name"]) == key_part.value),
                    None,
                )
                if ref_part is None:
                    continue

                rec = {
                    "frame_number":  fn,
                    "jersey_number": jersey,
                    "team":          team_id,
                    "team_name":     team_name,
                    "pos_x":         ref_part["position_x"],
                    "pos_y":         ref_part["position_y"],
                    "pos_z":         ref_part["position_z"],
                    **ball_rec,
                }
                records.append(rec)

        df = pd.DataFrame(records)
        if not df.empty:
            df["frame_number"]  = df["frame_number"].astype("int32")
            df["jersey_number"] = df["jersey_number"].astype("int8")
            df["team"]          = df["team"].astype("int8")
        return df

    @property
    def frame_count(self) -> int:
        self._load()
        return self._table.num_rows
