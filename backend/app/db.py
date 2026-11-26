"""
Nidaan-Rx v2.0 — SQLite Persistence for Users, Revoked Tokens, Profiles & Private Triage Cases.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("data/nidaan_v2_cases.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    username TEXT UNIQUE,
    full_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    weight_updated_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    profile_completed INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS revoked_tokens (
    jti TEXT PRIMARY KEY,
    expires_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS assessment_sessions (
    case_id TEXT PRIMARY KEY,
    user_id TEXT,
    created_at TEXT NOT NULL,
    backbone TEXT NOT NULL,
    view_count INTEGER NOT NULL,
    top_condition_id TEXT,
    top_condition_name TEXT,
    top_confidence REAL,
    is_ood INTEGER NOT NULL DEFAULT 0,
    ood_score REAL,
    escalation_hard_stop INTEGER NOT NULL DEFAULT 0,
    profile_json TEXT NOT NULL,
    case_context_json TEXT NOT NULL,
    inference_json TEXT NOT NULL,
    assessment_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_followups (
    followup_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    followup_day INTEGER NOT NULL,
    followup_trend TEXT NOT NULL,
    itch_score INTEGER,
    pain_burning_score INTEGER,
    notes TEXT NOT NULL DEFAULT '',
    escalated INTEGER NOT NULL DEFAULT 0,
    escalation_reason TEXT,
    assessment_json TEXT NOT NULL,
    FOREIGN KEY (case_id) REFERENCES assessment_sessions(case_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_case_followups_case ON case_followups(case_id, created_at);
"""


class CaseRepository:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(assessment_sessions)").fetchall()
            }
            if "user_id" not in cols:
                conn.execute("ALTER TABLE assessment_sessions ADD COLUMN user_id TEXT")
            prof_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(user_profiles)").fetchall()
            }
            if "profile_completed" not in prof_cols:
                conn.execute(
                    "ALTER TABLE user_profiles ADD COLUMN profile_completed INTEGER NOT NULL DEFAULT 1"
                )
            user_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(users)").fetchall()
            }
            if "username" not in user_cols:
                conn.execute("ALTER TABLE users ADD COLUMN username TEXT")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username)")

    # ---------------- Token Revocation ----------------
    def revoke_token(self, jti: str, expires_at: int) -> None:
        now = int(time.time())
        with self._connect() as conn:
            conn.execute("DELETE FROM revoked_tokens WHERE expires_at < ?", (now,))
            conn.execute(
                "INSERT OR REPLACE INTO revoked_tokens (jti, expires_at) VALUES (?, ?)",
                (jti, int(expires_at)),
            )

    def is_token_revoked(self, jti: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM revoked_tokens WHERE jti = ?",
                (jti,),
            ).fetchone()
        return row is not None

    # ---------------- Users & Profiles ----------------
    def create_user(
        self,
        *,
        email: str,
        full_name: str,
        password_hash: str,
        username: str | None = None,
    ) -> dict[str, Any]:
        import re as _re
        uid = f"usr_{uuid.uuid4().hex[:12]}"
        created_at = datetime.now(timezone.utc).isoformat()
        clean_email = email.strip().lower()
        clean_name = full_name.strip()
        if username and username.strip():
            clean_username = username.strip().lower()
        else:
            base = _re.sub(r"[^a-z0-9_.-]+", "", clean_email.split("@")[0]) or "user"
            clean_username = f"{base[:20]}_{uid[-4:]}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users (user_id, email, username, full_name, password_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (uid, clean_email, clean_username, clean_name, password_hash, created_at),
            )
        return {
            "user_id": uid,
            "email": clean_email,
            "username": clean_username,
            "full_name": clean_name,
            "created_at": created_at,
        }

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?",
                (email.strip().lower(),),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE LOWER(username) = ?",
                (username.strip().lower(),),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id, email, username, full_name, created_at FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_user_with_password_by_id(self, user_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id, email, full_name, password_hash, created_at FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def update_user_password(self, user_id: str, password_hash: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE users SET password_hash = ? WHERE user_id = ?",
                (password_hash, user_id),
            )
            return cur.rowcount > 0

    def get_user_profile(self, user_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile_json, weight_updated_at, updated_at, profile_completed FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "profile": json.loads(row["profile_json"]),
            "weight_updated_at": row["weight_updated_at"],
            "updated_at": row["updated_at"],
            "profile_completed": bool(row["profile_completed"]),
        }

    def upsert_user_profile(
        self,
        user_id: str,
        profile_dict: dict[str, Any],
        force_weight_timestamp: bool = False,
        mark_completed: bool | None = None,
    ) -> dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat()
        existing = self.get_user_profile(user_id)

        if existing is None or force_weight_timestamp:
            weight_updated_at = now_iso
        else:
            prev_weight = existing["profile"].get("weight_kg")
            new_weight = profile_dict.get("weight_kg")
            weight_updated_at = now_iso if prev_weight != new_weight else existing["weight_updated_at"]

        if mark_completed is not None:
            completed_int = 1 if mark_completed else 0
        elif existing is not None:
            completed_int = 1 if existing.get("profile_completed") else 0
        else:
            completed_int = 0

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_profiles (user_id, profile_json, weight_updated_at, updated_at, profile_completed)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    weight_updated_at = excluded.weight_updated_at,
                    updated_at = excluded.updated_at,
                    profile_completed = excluded.profile_completed
                """,
                (user_id, json.dumps(profile_dict, default=str), weight_updated_at, now_iso, completed_int),
            )
        return {
            "profile": profile_dict,
            "weight_updated_at": weight_updated_at,
            "updated_at": now_iso,
            "profile_completed": bool(completed_int),
        }

    # ---------------- Private User Cases ----------------
    def save_assessment(
        self,
        *,
        backbone: str,
        view_count: int,
        profile_dict: dict[str, Any],
        case_dict: dict[str, Any],
        inference_dict: dict[str, Any],
        assessment_dict: dict[str, Any],
        case_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        cid = case_id or f"ndx_{uuid.uuid4().hex[:12]}"
        created_at = datetime.now(timezone.utc).isoformat()
        top_matches = assessment_dict.get("top_matches") or []
        top = top_matches[0] if top_matches else {}

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO assessment_sessions (
                    case_id, user_id, created_at, backbone, view_count,
                    top_condition_id, top_condition_name, top_confidence,
                    is_ood, ood_score, escalation_hard_stop,
                    profile_json, case_context_json, inference_json, assessment_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cid,
                    user_id,
                    created_at,
                    backbone,
                    int(view_count),
                    top.get("condition_id"),
                    top.get("display_name"),
                    float(top.get("posterior_prob", 0.0)) if top else 0.0,
                    1 if inference_dict.get("is_ood") else 0,
                    float(inference_dict.get("ood_score", 0.0)),
                    1 if assessment_dict.get("escalation_hard_stop") else 0,
                    json.dumps(profile_dict, default=str),
                    json.dumps(case_dict, default=str),
                    json.dumps(inference_dict, default=str),
                    json.dumps(assessment_dict, default=str),
                ),
            )
        return cid

    def list_recent(self, limit: int = 20, user_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT a.case_id, a.user_id, a.created_at, a.backbone, a.view_count,
                       a.top_condition_id, a.top_condition_name, a.top_confidence,
                       a.is_ood, a.ood_score, a.escalation_hard_stop,
                       (SELECT COUNT(*) FROM case_followups f
                        WHERE f.case_id = a.case_id AND f.user_id = a.user_id) AS followup_count,
                       (SELECT f.followup_day FROM case_followups f
                        WHERE f.case_id = a.case_id AND f.user_id = a.user_id
                        ORDER BY f.followup_day DESC, f.created_at DESC LIMIT 1) AS latest_followup_day,
                       (SELECT f.followup_trend FROM case_followups f
                        WHERE f.case_id = a.case_id AND f.user_id = a.user_id
                        ORDER BY f.followup_day DESC, f.created_at DESC LIMIT 1) AS latest_followup_trend,
                       (SELECT f.escalated FROM case_followups f
                        WHERE f.case_id = a.case_id AND f.user_id = a.user_id
                        ORDER BY f.followup_day DESC, f.created_at DESC LIMIT 1) AS longitudinal_escalated
                FROM assessment_sessions a
                WHERE a.user_id = ?
                ORDER BY a.created_at DESC
                LIMIT ?
                """,
                (user_id, int(limit)),
            ).fetchall()
        return [
            {
                **dict(r),
                "is_ood": bool(r["is_ood"]),
                "escalation_hard_stop": bool(r["escalation_hard_stop"]),
                "followup_count": int(r["followup_count"] or 0),
                "longitudinal_escalated": bool(r["longitudinal_escalated"]),
            }
            for r in rows
        ]

    def get_case(self, case_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM assessment_sessions WHERE case_id = ? AND user_id = ?",
                (case_id, user_id),
            ).fetchone()
        if row is None:
            return None
        followups = self.list_followups(case_id, user_id=row["user_id"])
        return {
            "case_id": row["case_id"],
            "user_id": row["user_id"],
            "created_at": row["created_at"],
            "backbone": row["backbone"],
            "view_count": row["view_count"],
            "is_ood": bool(row["is_ood"]),
            "ood_score": row["ood_score"],
            "escalation_hard_stop": bool(row["escalation_hard_stop"]),
            "profile": json.loads(row["profile_json"]),
            "case_context": json.loads(row["case_context_json"]),
            "inference": json.loads(row["inference_json"]),
            "assessment": json.loads(row["assessment_json"]),
            "followups": followups,
        }

    def get_max_followup_day(self, case_id: str, user_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(followup_day), 0) AS max_day FROM case_followups WHERE case_id = ? AND user_id = ?",
                (case_id, user_id),
            ).fetchone()
        return int(row["max_day"]) if row is not None else 0

    def get_latest_followup_with_assessment(self, case_id: str, user_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT followup_id, case_id, created_at, followup_day, followup_trend,
                       itch_score, pain_burning_score, notes, escalated, escalation_reason, assessment_json
                FROM case_followups
                WHERE case_id = ? AND user_id = ?
                ORDER BY followup_day DESC, created_at DESC
                LIMIT 1
                """,
                (case_id, user_id),
            ).fetchone()
        if row is None:
            return None
        return {
            **dict(row),
            "escalated": bool(row["escalated"]),
            "assessment": json.loads(row["assessment_json"]),
        }

    def delete_followup(self, case_id: str, followup_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM case_followups WHERE followup_id = ? AND case_id = ? AND user_id = ?",
                (followup_id, case_id, user_id),
            )
            return cur.rowcount > 0

    def update_case_state(
        self,
        *,
        case_id: str,
        user_id: str,
        case_context_dict: dict[str, Any],
        assessment_dict: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE assessment_sessions
                SET case_context_json = ?,
                    assessment_json = ?,
                    escalation_hard_stop = ?
                WHERE case_id = ? AND user_id = ?
                """,
                (
                    json.dumps(case_context_dict, default=str),
                    json.dumps(assessment_dict, default=str),
                    1 if assessment_dict.get("escalation_hard_stop") else 0,
                    case_id,
                    user_id,
                ),
            )

    def add_followup(
        self,
        *,
        case_id: str,
        user_id: str,
        followup_day: int,
        followup_trend: str,
        itch_score: int | None,
        pain_burning_score: int | None,
        notes: str,
        escalated: bool,
        escalation_reason: str | None,
        updated_case_dict: dict[str, Any],
        updated_assessment_dict: dict[str, Any],
    ) -> dict[str, Any]:
        fid = f"fu_{uuid.uuid4().hex[:12]}"
        created_at = datetime.now(timezone.utc).isoformat()
        clean_notes = (notes or "").strip()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO case_followups (
                    followup_id, case_id, user_id, created_at,
                    followup_day, followup_trend, itch_score, pain_burning_score,
                    notes, escalated, escalation_reason, assessment_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fid,
                    case_id,
                    user_id,
                    created_at,
                    int(followup_day),
                    followup_trend,
                    int(itch_score) if itch_score is not None else None,
                    int(pain_burning_score) if pain_burning_score is not None else None,
                    clean_notes,
                    1 if escalated else 0,
                    escalation_reason,
                    json.dumps(updated_assessment_dict, default=str),
                ),
            )
            conn.execute(
                """
                UPDATE assessment_sessions
                SET case_context_json = ?,
                    assessment_json = ?,
                    escalation_hard_stop = ?
                WHERE case_id = ? AND user_id = ?
                """,
                (
                    json.dumps(updated_case_dict, default=str),
                    json.dumps(updated_assessment_dict, default=str),
                    1 if updated_assessment_dict.get("escalation_hard_stop") else 0,
                    case_id,
                    user_id,
                ),
            )
        return {
            "followup_id": fid,
            "case_id": case_id,
            "created_at": created_at,
            "followup_day": int(followup_day),
            "followup_trend": followup_trend,
            "itch_score": int(itch_score) if itch_score is not None else None,
            "pain_burning_score": int(pain_burning_score) if pain_burning_score is not None else None,
            "notes": clean_notes,
            "escalated": bool(escalated),
            "escalation_reason": escalation_reason,
        }

    def list_followups(self, case_id: str, user_id: str | None) -> list[dict[str, Any]]:
        if not user_id:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT followup_id, case_id, created_at, followup_day, followup_trend,
                       itch_score, pain_burning_score, notes, escalated, escalation_reason
                FROM case_followups
                WHERE case_id = ? AND user_id = ?
                ORDER BY followup_day ASC, created_at ASC
                """,
                (case_id, user_id),
            ).fetchall()
        return [
            {
                **dict(r),
                "escalated": bool(r["escalated"]),
            }
            for r in rows
        ]

    def delete_case(self, case_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM case_followups WHERE case_id = ? AND user_id = ?",
                (case_id, user_id),
            )
            cur = conn.execute(
                "DELETE FROM assessment_sessions WHERE case_id = ? AND user_id = ?",
                (case_id, user_id),
            )
            return cur.rowcount > 0
