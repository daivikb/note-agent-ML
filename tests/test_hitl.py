"""
test_hitl.py

Offline tests for the HITL feedback store and few-shot prompt injection.
No API key required — all tests use local SQLite.

Run: python test_hitl.py
"""

import os
import tempfile

# Use a temporary database for tests
_test_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["NOTE_AGENT_FEEDBACK_DB"] = _test_db.name

from ml.feedback import (
    init_feedback_db,
    log_extraction,
    submit_review,
    get_pending_reviews,
    get_pending_count,
    get_reviewed_count,
    get_review_stats,
    get_few_shot_examples,
    format_few_shot_block,
)
from ml.extraction import ExtractedObject


def _make_objects():
    """Create sample ExtractedObject instances for testing."""
    return [
        ExtractedObject(
            id="obj_001",
            type="Idea",
            canonical_text="Launch new product by March",
            confidence=0.9,
        ),
        ExtractedObject(
            id="obj_002",
            type="Claim",
            canonical_text="Budget is approved by the board",
            confidence=0.85,
        ),
        ExtractedObject(
            id="obj_003",
            type="Question",
            canonical_text="What is the go-to-market timeline?",
            confidence=0.8,
        ),
    ]


def test_feedback_store_crud():
    """Test basic CRUD operations on the feedback store."""
    print("\n── Test: Feedback Store CRUD ──")

    init_feedback_db()
    objects = _make_objects()

    # Log extraction
    count = log_extraction("note_test_001", objects)
    assert count == 3, f"Expected 3 logged items, got {count}"
    print("  ✓ Logged 3 objects as pending")

    # Query pending
    pending = get_pending_reviews("note_test_001")
    assert len(pending) == 3, f"Expected 3 pending, got {len(pending)}"
    assert all(r["status"] == "pending" for r in pending)
    print("  ✓ All 3 objects are pending")

    # Accept one
    submit_review(pending[0]["id"], "accepted")
    assert get_pending_count() == 2
    assert get_reviewed_count() == 1
    print("  ✓ Accepted obj_001, counts updated")

    # Correct one (type change)
    submit_review(pending[1]["id"], "corrected", corrected_type="Assumption")
    stats = get_review_stats()
    assert stats.get("corrected") == 1
    print("  ✓ Corrected obj_002 (Claim → Assumption)")

    # Reject one
    submit_review(pending[2]["id"], "rejected")
    stats = get_review_stats()
    assert stats.get("rejected") == 1
    print("  ✓ Rejected obj_003")

    # No pending left
    assert get_pending_count() == 0
    print("  ✓ No pending reviews remain")


def test_few_shot_examples():
    """Test that corrected reviews produce valid few-shot examples."""
    print("\n── Test: Few-Shot Example Generation ──")

    examples = get_few_shot_examples(limit=10)
    # Should have at least the 1 correction + 1 accepted from above
    assert len(examples) >= 1, f"Expected at least 1 example, got {len(examples)}"
    print(f"  ✓ Retrieved {len(examples)} few-shot example(s)")

    # Find the correction
    corrections = [e for e in examples if e["original_type"] != e["corrected_type"]]
    assert len(corrections) >= 1, "Should have at least one correction example"
    c = corrections[0]
    assert c["original_type"] == "Claim"
    assert c["corrected_type"] == "Assumption"
    print(f"  ✓ Correction example: {c['original_type']} → {c['corrected_type']}")


def test_format_few_shot_block():
    """Test that the few-shot block formats correctly for prompt injection."""
    print("\n── Test: Few-Shot Block Formatting ──")

    # Empty case
    empty_block = format_few_shot_block([])
    assert empty_block == "", "Empty examples should produce empty block"
    print("  ✓ Empty examples → empty block (zero-shot fallback)")

    # With examples
    examples = get_few_shot_examples(limit=5)
    block = format_few_shot_block(examples)
    assert "Learn from these" in block
    assert "Incorrect:" in block or "Correct classification:" in block
    print(f"  ✓ Block generated ({len(block)} chars)")
    print(f"\n  ── Generated Block Preview ──")
    for line in block.split("\n")[:8]:
        print(f"    {line}")
    print(f"    ...")


def test_invalid_action():
    """Test that invalid review actions raise errors."""
    print("\n── Test: Invalid Action Validation ──")

    try:
        submit_review(999, "invalid_action")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  ✓ Invalid action raises ValueError")


def test_status_tracking():
    """Test review statistics are accurate."""
    print("\n── Test: Status Tracking ──")

    stats = get_review_stats()
    assert "accepted" in stats
    assert "corrected" in stats
    assert "rejected" in stats
    total = sum(stats.values())
    assert total == 3, f"Expected 3 total reviews, got {total}"
    print(f"  ✓ Stats: {dict(stats)}")
    print(f"  ✓ Total: {total}")


def main():
    print("=" * 60)
    print("  HITL FEEDBACK SYSTEM — OFFLINE TESTS")
    print("=" * 60)

    test_feedback_store_crud()
    test_few_shot_examples()
    test_format_few_shot_block()
    test_invalid_action()
    test_status_tracking()

    print("\n" + "=" * 60)
    print("  ✓ ALL HITL TESTS PASSED")
    print("=" * 60)

    # Clean up temp DB
    try:
        os.unlink(_test_db.name)
    except OSError:
        pass


if __name__ == "__main__":
    main()
