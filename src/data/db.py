from __future__ import annotations
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
import pandas as pd
DEFAULT_DB_PATH = os.path.join("data", "app.db")

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_conn(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
              user_id INTEGER PRIMARY KEY,
              skills_json TEXT,
              interests_json TEXT,
              experience TEXT,
              profile_text TEXT,
              created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS opportunities (
              opportunity_id INTEGER PRIMARY KEY,
              title TEXT NOT NULL,
              description TEXT NOT NULL,
              skills_required_json TEXT,
              category TEXT,
              location TEXT,
              opportunity_type TEXT,
              created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
              interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              opportunity_id INTEGER NOT NULL,
              event_type TEXT NOT NULL,
              weight REAL NOT NULL,
              ts TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(user_id),
              FOREIGN KEY(opportunity_id) REFERENCES opportunities(opportunity_id)
            )
            """
        )


def seed_from_csv(
    *,
    db_path: str = DEFAULT_DB_PATH,
    opportunities_csv: str = os.path.join("data", "opportunities.csv"),
    interactions_csv: str = os.path.join("data", "interactions.csv"),
) -> None:
    
    init_db(db_path)

    with get_conn(db_path) as conn:
        opp_count = conn.execute("SELECT COUNT(1) AS c FROM opportunities").fetchone()["c"]
        if opp_count == 0 and os.path.exists(opportunities_csv):
            opps = pd.read_csv(opportunities_csv).fillna("")
            for _, r in opps.iterrows():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO opportunities (
                      opportunity_id, title, description,
                      skills_required_json, category, location, opportunity_type,
                      created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(r["opportunity_id"]),
                        str(r.get("title", "")),
                        str(r.get("description", "")),
                        json.dumps([]),
                        "",
                        "",
                        "",
                        _utc_now_iso(),
                    ),
                )

        inter_count = conn.execute("SELECT COUNT(1) AS c FROM interactions").fetchone()["c"]
        if inter_count == 0 and os.path.exists(interactions_csv):
            inter = pd.read_csv(interactions_csv).fillna("")
            if "user_id" in inter.columns:
                for uid in sorted(set(inter["user_id"].astype(int).tolist())):
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO users (
                          user_id, skills_json, interests_json, experience, profile_text, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (int(uid), json.dumps([]), json.dumps([]), "", "", _utc_now_iso()),
                    )

            for _, r in inter.iterrows():
                uid = int(r["user_id"])
                oid = int(r["opportunity_id"])
                weight = float(r.get("interaction", 1.0) or 1.0)
                conn.execute(
                    """
                    INSERT INTO interactions (user_id, opportunity_id, event_type, weight, ts)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (uid, oid, "implicit", weight, _utc_now_iso()),
                )

def fetch_df(db_path: str, query: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    with get_conn(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def fetch_opportunities(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    df = fetch_df(
        db_path,
        """
        SELECT opportunity_id, title, description, skills_required_json, category, location, opportunity_type
        FROM opportunities
        ORDER BY opportunity_id ASC
        """,
    )
    if df.empty:
        return df
    df = df.fillna("")
    return df


def fetch_users(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    df = fetch_df(
        db_path,
        """
        SELECT user_id, skills_json, interests_json, experience, profile_text
        FROM users
        ORDER BY user_id ASC
        """,
    )
    return df.fillna("")


def fetch_interactions(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    df = fetch_df(
        db_path,
        """
        SELECT user_id, opportunity_id, event_type, weight, ts
        FROM interactions
        """,
    )
    return df.fillna("")


def upsert_user(
    *,
    db_path: str = DEFAULT_DB_PATH,
    user_id: int,
    skills: Optional[list[str]] = None,
    interests: Optional[list[str]] = None,
    experience: str = "",
    profile_text: str = "",
) -> None:
    init_db(db_path)
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO users (user_id, skills_json, interests_json, experience, profile_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
              skills_json=excluded.skills_json,
              interests_json=excluded.interests_json,
              experience=excluded.experience,
              profile_text=excluded.profile_text
            """,
            (
                int(user_id),
                json.dumps(skills or []),
                json.dumps(interests or []),
                experience or "",
                profile_text or "",
                _utc_now_iso(),
            ),
        )

def insert_opportunity(
    *,
    db_path: str = DEFAULT_DB_PATH,
    opportunity_id: int,
    title: str,
    description: str,
    skills_required: Optional[list[str]] = None,
    category: str = "",
    location: str = "",
    opportunity_type: str = "",
) -> None:
    init_db(db_path)
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO opportunities (
              opportunity_id, title, description, skills_required_json,
              category, location, opportunity_type, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(opportunity_id),
                title,
                description,
                json.dumps(skills_required or []),
                category,
                location,
                opportunity_type,
                _utc_now_iso(),
            ),
        )

def insert_interaction(
    *,
    db_path: str = DEFAULT_DB_PATH,
    user_id: int,
    opportunity_id: int,
    event_type: str,
    weight: float,
    ts: Optional[str] = None,
) -> None:
    init_db(db_path)
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO users (user_id, skills_json, interests_json, experience, profile_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (int(user_id), json.dumps([]), json.dumps([]), "", "", _utc_now_iso()),
        )
        conn.execute(
            """
            INSERT INTO interactions (user_id, opportunity_id, event_type, weight, ts)
            VALUES (?, ?, ?, ?, ?)
            """,
            (int(user_id), int(opportunity_id), event_type, float(weight), ts or _utc_now_iso()),
        )

