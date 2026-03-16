"""
feedback.py

SQLite-backed feedback store for the Human-in-the-Loop (HITL) system.
Tracks human reviews of extracted objects and provides corrected examples
for dynamic few-shot prompt injection.
"""

import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional


def _feedback_db_path() -> str:
    """Resolve the path to the feedback database."""
    return os.environ.get("NOTE_AGENT_FEEDBACK_DB", "feedback.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_feedback_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_feedback_db() -> None:
    """Create the human_reviews table if it does not exist."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS human_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_id TEXT NOT NULL,
                note_id TEXT NOT NULL,
                original_type TEXT NOT NULL,
                corrected_type TEXT,
                original_text TEXT NOT NULL,
                corrected_text TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                reviewed_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_reviews_status
            ON human_reviews(status)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_reviews_note
            ON human_reviews(note_id)
            """
        )


# ── Logging ──────────────────────────────────────────────────────────────────

def log_extraction(note_id: str, objects: list) -> int:
    """
    Log all extracted objects as 'pending' review items.
    Called automatically after each extraction run.

    Args:
        note_id: The source note identifier.
        objects: List of ExtractedObject instances.

    Returns:
        Number of items logged.
    """
    init_feedback_db()
    rows = [
        (obj.id, note_id, obj.type, obj.canonical_text)
        for obj in objects
    ]
    if not rows:
        return 0

    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO human_reviews (object_id, note_id, original_type, original_text, status)
            VALUES (?, ?, ?, ?, 'pending')
            """,
            rows,
        )
        conn.commit()
    return len(rows)


# ── Review Actions ───────────────────────────────────────────────────────────

def submit_review(
    review_id: int,
    action: str,
    corrected_type: Optional[str] = None,
    corrected_text: Optional[str] = None,
) -> None:
    """
    Record a human review decision for an extracted object.

    Args:
        review_id: Primary key in human_reviews table.
        action: One of 'accepted', 'rejected', 'corrected'.
        corrected_type: The corrected object type (only for 'corrected').
        corrected_text: The corrected canonical text (only for 'corrected').
    """
    if action not in ("accepted", "rejected", "corrected"):
        raise ValueError(f"Invalid action: {action}. Must be 'accepted', 'rejected', or 'corrected'.")

    init_feedback_db()
    now = datetime.now().isoformat()

    with _connect() as conn:
        conn.execute(
            """
            UPDATE human_reviews
            SET status = ?,
                corrected_type = ?,
                corrected_text = ?,
                reviewed_at = ?
            WHERE id = ?
            """,
            (action, corrected_type, corrected_text, now, review_id),
        )
        conn.commit()


# ── Queries ──────────────────────────────────────────────────────────────────

def get_pending_reviews(note_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return all objects awaiting human review, optionally filtered by note."""
    init_feedback_db()
    with _connect() as conn:
        if note_id:
            rows = conn.execute(
                "SELECT * FROM human_reviews WHERE status = 'pending' AND note_id = ? ORDER BY id",
                (note_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM human_reviews WHERE status = 'pending' ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]


def get_reviewed_count() -> int:
    """Return the count of reviewed (non-pending) items."""
    init_feedback_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM human_reviews WHERE status != 'pending'"
        ).fetchone()
        return row["cnt"]


def get_pending_count() -> int:
    """Return the count of pending items."""
    init_feedback_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM human_reviews WHERE status = 'pending'"
        ).fetchone()
        return row["cnt"]


def get_review_stats() -> Dict[str, int]:
    """Return counts grouped by status."""
    init_feedback_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM human_reviews GROUP BY status"
        ).fetchall()
        return {r["status"]: r["cnt"] for r in rows}


# ── Few-Shot Example Generation ─────────────────────────────────────────────

def get_few_shot_examples(limit: int = 5) -> List[Dict[str, str]]:
    """
    Retrieve corrected and accepted reviews to use as few-shot examples in prompts.
    Prioritizes corrections (where human changed the type) since those
    are most informative for improving the LLM.

    Args:
        limit: Maximum number of examples to return.

    Returns:
        List of dicts with keys: original_text, original_type, corrected_type, corrected_text, status
    """
    init_feedback_db()
    with _connect() as conn:
        # Prioritize corrections (type changes are most valuable)
        rows = conn.execute(
            """
            SELECT original_text, original_type, corrected_type, corrected_text,
                   'corrected' as status
            FROM human_reviews
            WHERE status = 'corrected'
              AND corrected_type IS NOT NULL
            ORDER BY reviewed_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        examples = [dict(r) for r in rows]

        # If we have fewer than limit, also include accepted examples
        # (these confirm correct classifications)
        remaining = limit - len(examples)
        if remaining > 0:
            accepted_rows = conn.execute(
                """
                SELECT original_text, original_type,
                       original_type as corrected_type,
                       original_text as corrected_text,
                       'accepted' as status
                FROM human_reviews
                WHERE status = 'accepted'
                ORDER BY reviewed_at DESC
                LIMIT ?
                """,
                (remaining,),
            ).fetchall()
            examples.extend(dict(r) for r in accepted_rows)

        # Also include rejected examples as negative examples
        rejected_rows = conn.execute(
            """
            SELECT original_text, original_type, original_type as corrected_type,
                   original_text as corrected_text, 'rejected' as status
            FROM human_reviews
            WHERE status = 'rejected'
            ORDER BY reviewed_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        examples.extend(dict(r) for r in rejected_rows)

        return examples


def format_few_shot_block(examples: List[Dict[str, str]]) -> str:
    """
    Format few-shot examples into a prompt block for injection into
    the extraction prompt.

    Returns empty string if no examples are available.
    """
    if not examples:
        return ""

    lines = [
        "Learn from these previous human-verified corrections to improve your classifications:\n"
    ]

    for i, ex in enumerate(examples, 1):
        original_type = ex["original_type"]
        corrected_type = ex.get("corrected_type") or original_type
        text = ex.get("corrected_text") or ex["original_text"]
        status = ex.get("status", "")

        if status == "rejected":
            # This was rejected — tell the LLM not to extract fragments like this
            lines.append(f"Example {i}:")
            lines.append(f'  Text: "{text}"')
            lines.append(f"  DO NOT extract this — it is a noisy fragment, not a valid knowledge object.")
        elif original_type != corrected_type:
            # This was a correction — show the mistake and fix
            lines.append(f"Example {i}:")
            lines.append(f'  Text: "{text}"')
            lines.append(f"  Incorrect: {original_type} → Correct: {corrected_type}")
        else:
            # This was accepted — confirm the classification
            lines.append(f"Example {i}:")
            lines.append(f'  Text: "{text}"')
            lines.append(f"  Correct classification: {corrected_type}")
        lines.append("")

    return "\n".join(lines)
