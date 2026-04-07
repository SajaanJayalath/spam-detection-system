from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid


class HistoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    text TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL
                )
                """
            )
            conn.commit()

    def add(self, *, text: str, prediction: str, confidence: float, timestamp: Optional[str] = None) -> str:
        item_id = uuid.uuid4().hex
        ts = timestamp or datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO history (id, timestamp, text, prediction, confidence) VALUES (?, ?, ?, ?, ?)",
                (item_id, ts, text, prediction, float(confidence)),
            )
            conn.commit()
        return item_id

    def list(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        q = "SELECT id, timestamp, text, prediction, confidence FROM history ORDER BY datetime(timestamp) DESC"
        if limit is not None:
            q += " LIMIT ?"
        with self._connect() as conn:
            cur = conn.execute(q, (() if limit is None else (int(limit),)))
            rows = [dict(r) for r in cur.fetchall()]
        return rows

    def get(self, item_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT id, timestamp, text, prediction, confidence FROM history WHERE id = ?",
                (item_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def update(self, item_id: str, *, prediction: Optional[str] = None, confidence: Optional[float] = None) -> Optional[Dict[str, Any]]:
        fields: List[str] = []
        params: List[Any] = []
        if prediction is not None:
            fields.append("prediction = ?")
            params.append(prediction)
        if confidence is not None:
            fields.append("confidence = ?")
            params.append(float(confidence))
        if not fields:
            return self.get(item_id)
        params.append(item_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE history SET {', '.join(fields)} WHERE id = ?", tuple(params))
            conn.commit()
        return self.get(item_id)

    def delete(self, item_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM history WHERE id = ?", (item_id,))
            conn.commit()
            return cur.rowcount > 0


# Singleton store at backend/data/history.db
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
HISTORY_DB = DATA_DIR / "history.db"
HISTORY_STORE = HistoryStore(HISTORY_DB)

