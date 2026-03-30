from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------- File fallback configuration ----------
BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_DIR = BASE_DIR / "outputs" / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_FILE = HISTORY_DIR / "message_history.json"

# ---------- MySQL configuration ----------
MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DATABASE = "sms_classifier"
MYSQL_TABLE = "message_analyses"


def _build_record(sender: str, message: str, ensemble_result: dict, nn_result: dict) -> dict[str, Any]:
    return {
        "sender": sender or "unknown",
        "message": message,
        "ensemble_prediction": ensemble_result.get("prediction"),
        "ensemble_confidence": ensemble_result.get("confidence"),
        "nn_prediction": nn_result.get("prediction"),
        "nn_confidence": nn_result.get("confidence"),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _save_to_json(record: dict[str, Any]) -> None:
    if HISTORY_FILE.exists():
        try:
            existing = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    else:
        existing = []

    existing.append(record)
    HISTORY_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


def _get_mysql_connector():
    try:
        import mysql.connector
        return mysql.connector
    except Exception:
        return None


def _ensure_database_and_table() -> bool:
    mysql_connector = _get_mysql_connector()
    if mysql_connector is None:
        return False

    conn = None
    cursor = None
    try:
        # Connect without specifying the database first
        conn = mysql_connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
        )
        cursor = conn.cursor()

        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}`")
        cursor.execute(f"USE `{MYSQL_DATABASE}`")

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS `{MYSQL_TABLE}` (
                id INT AUTO_INCREMENT PRIMARY KEY,
                sender VARCHAR(255) NOT NULL,
                message TEXT NOT NULL,
                ensemble_prediction VARCHAR(100),
                ensemble_confidence DECIMAL(8,4),
                nn_prediction VARCHAR(100),
                nn_confidence DECIMAL(8,4),
                created_at DATETIME NOT NULL
            )
            """
        )

        conn.commit()
        return True

    except Exception:
        return False

    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _save_to_mysql(record: dict[str, Any]) -> bool:
    mysql_connector = _get_mysql_connector()
    if mysql_connector is None:
        return False

    if not _ensure_database_and_table():
        return False

    conn = None
    cursor = None
    try:
        conn = mysql_connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
        )
        cursor = conn.cursor()

        cursor.execute(
            f"""
            INSERT INTO `{MYSQL_TABLE}` (
                sender,
                message,
                ensemble_prediction,
                ensemble_confidence,
                nn_prediction,
                nn_confidence,
                created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                record["sender"],
                record["message"],
                record["ensemble_prediction"],
                record["ensemble_confidence"],
                record["nn_prediction"],
                record["nn_confidence"],
                record["created_at"],
            ),
        )

        conn.commit()
        return True

    except Exception:
        return False

    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def save_analysis_history(
    sender: str,
    message: str,
    ensemble_result: dict,
    nn_result: dict,
) -> str:
    """
    Save analysis history.
    Returns:
        'mysql' if saved to MySQL
        'json' if saved to fallback JSON file
    """
    record = _build_record(
        sender=sender,
        message=message,
        ensemble_result=ensemble_result,
        nn_result=nn_result,
    )

    if _save_to_mysql(record):
        return "mysql"

    _save_to_json(record)
    return "json"


def load_analysis_history(limit: int | None = None) -> list[dict[str, Any]]:
    """
    Load history from MySQL if available, otherwise from JSON.
    Most recent records come first.
    """
    mysql_connector = _get_mysql_connector()

    if mysql_connector is not None and _ensure_database_and_table():
        conn = None
        cursor = None
        try:
            conn = mysql_connector.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE,
            )
            cursor = conn.cursor(dictionary=True)

            query = f"""
                SELECT
                    sender,
                    message,
                    ensemble_prediction,
                    ensemble_confidence,
                    nn_prediction,
                    nn_confidence,
                    created_at
                FROM `{MYSQL_TABLE}`
                ORDER BY created_at DESC, id DESC
            """

            if limit is not None and limit > 0:
                query += " LIMIT %s"
                cursor.execute(query, (limit,))
            else:
                cursor.execute(query)

            rows = cursor.fetchall() or []
            for row in rows:
                if row.get("created_at") is not None:
                    row["created_at"] = str(row["created_at"])
            return rows

        except Exception:
            pass

        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    if not HISTORY_FILE.exists():
        return []

    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        data = list(reversed(data))
        if limit is not None and limit > 0:
            return data[:limit]
        return data
    except Exception:
        return []


def clear_json_history() -> bool:
    """
    Optional helper to clear only the JSON fallback history file.
    Does not affect MySQL.
    """
    try:
        HISTORY_FILE.write_text("[]", encoding="utf-8")
        return True
    except Exception:
        return False