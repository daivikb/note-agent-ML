"""
hitl.py

Human-in-the-Loop CLI review tool for extracted knowledge objects.
Lets users review, accept, reject, or correct LLM extractions interactively.

Usage:
    python -m ml.hitl                    # Review all pending objects
    python -m ml.hitl --note-id note_1   # Review objects from a specific note
"""

import argparse
import sys

from ml.feedback import (
    get_pending_reviews,
    get_review_stats,
    init_feedback_db,
    submit_review,
)
from ml.extraction import OBJECT_TYPE_DEFINITIONS


# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


VALID_TYPES = list(OBJECT_TYPE_DEFINITIONS.keys())


def _print_object(idx: int, total: int, review: dict) -> None:
    """Pretty-print a pending review item."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}─── Object {idx}/{total} ───{Colors.END}")
    print(f"  {Colors.BOLD}ID:{Colors.END}         {review['object_id']}")
    print(f"  {Colors.BOLD}Note:{Colors.END}       {review['note_id']}")
    print(f"  {Colors.BOLD}Type:{Colors.END}       {Colors.CYAN}{review['original_type']}{Colors.END}")
    print(f"  {Colors.BOLD}Text:{Colors.END}       \"{review['original_text']}\"")
    print(f"  {Colors.BOLD}Created:{Colors.END}    {review['created_at']}")


def _prompt_type_edit(current_type: str) -> str:
    """Prompt user to select a corrected type."""
    print(f"\n  Available types:")
    for i, t in enumerate(VALID_TYPES, 1):
        marker = " ← current" if t == current_type else ""
        print(f"    {i}. {t}{marker}")

    while True:
        choice = input(f"  Select type (1-{len(VALID_TYPES)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(VALID_TYPES):
                return VALID_TYPES[idx]
        except ValueError:
            # Also accept type name directly
            if choice in VALID_TYPES:
                return choice
        print(f"  {Colors.RED}Invalid choice. Try again.{Colors.END}")


def _prompt_text_edit(current_text: str) -> str:
    """Prompt user to enter corrected text."""
    print(f"\n  Current text: \"{current_text}\"")
    new_text = input("  Enter corrected text: ").strip()
    return new_text if new_text else current_text


def review_pending(note_id: str = None) -> None:
    """Interactive review loop for pending extractions."""
    init_feedback_db()

    pending = get_pending_reviews(note_id)
    if not pending:
        print(f"\n{Colors.GREEN}✓ No pending reviews.{Colors.END}")
        if note_id:
            print(f"  (filtered by note_id: {note_id})")
        return

    total = len(pending)
    print(f"\n{Colors.HEADER}{Colors.BOLD}══════════════════════════════════════════{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}  HITL Review: {total} objects pending{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}══════════════════════════════════════════{Colors.END}")

    counts = {"accepted": 0, "rejected": 0, "corrected": 0, "skipped": 0}

    for idx, review in enumerate(pending, 1):
        _print_object(idx, total, review)

        print(f"\n  {Colors.BOLD}Actions:{Colors.END} "
              f"[{Colors.GREEN}a{Colors.END}]ccept  "
              f"[{Colors.RED}r{Colors.END}]eject  "
              f"[{Colors.YELLOW}e{Colors.END}]dit type  "
              f"[{Colors.CYAN}t{Colors.END}]ext edit  "
              f"[s]kip  "
              f"[q]uit")

        while True:
            choice = input(f"  → ").strip().lower()

            if choice in ("a", "accept"):
                submit_review(review["id"], "accepted")
                counts["accepted"] += 1
                print(f"  {Colors.GREEN}✓ Accepted{Colors.END}")
                break

            elif choice in ("r", "reject"):
                submit_review(review["id"], "rejected")
                counts["rejected"] += 1
                print(f"  {Colors.RED}✗ Rejected{Colors.END}")
                break

            elif choice in ("e", "edit"):
                new_type = _prompt_type_edit(review["original_type"])
                submit_review(
                    review["id"], "corrected",
                    corrected_type=new_type,
                )
                counts["corrected"] += 1
                print(f"  {Colors.YELLOW}✎ Corrected type: {review['original_type']} → {new_type}{Colors.END}")
                break

            elif choice in ("t", "text"):
                new_text = _prompt_text_edit(review["original_text"])
                submit_review(
                    review["id"], "corrected",
                    corrected_type=review["original_type"],
                    corrected_text=new_text,
                )
                counts["corrected"] += 1
                print(f"  {Colors.YELLOW}✎ Text corrected{Colors.END}")
                break

            elif choice in ("s", "skip"):
                counts["skipped"] += 1
                print(f"  ⊘ Skipped")
                break

            elif choice in ("q", "quit"):
                print(f"\n  Exiting review early.")
                _print_summary(counts)
                return

            else:
                print(f"  {Colors.RED}Invalid choice. Use a/r/e/t/s/q.{Colors.END}")

    _print_summary(counts)


def _print_summary(counts: dict) -> None:
    """Print a summary of the review session."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}── Review Summary ──{Colors.END}")
    print(f"  {Colors.GREEN}Accepted:{Colors.END}  {counts['accepted']}")
    print(f"  {Colors.YELLOW}Corrected:{Colors.END} {counts['corrected']}")
    print(f"  {Colors.RED}Rejected:{Colors.END}  {counts['rejected']}")
    print(f"  Skipped:   {counts['skipped']}")

    # Show overall stats
    stats = get_review_stats()
    total_reviewed = sum(v for k, v in stats.items() if k != "pending")
    total_pending = stats.get("pending", 0)
    print(f"\n  {Colors.BOLD}Overall:{Colors.END} {total_reviewed} reviewed, {total_pending} pending")


def main():
    parser = argparse.ArgumentParser(
        description="Human-in-the-Loop review for extracted knowledge objects"
    )
    parser.add_argument(
        "--note-id",
        type=str,
        default=None,
        help="Filter reviews to a specific note ID",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show review statistics and exit",
    )
    args = parser.parse_args()

    if args.stats:
        init_feedback_db()
        stats = get_review_stats()
        print(f"\n{Colors.BOLD}Review Statistics:{Colors.END}")
        for status, count in stats.items():
            print(f"  {status}: {count}")
        total = sum(stats.values())
        print(f"  ─────────")
        print(f"  total: {total}")
        return

    review_pending(note_id=args.note_id)


if __name__ == "__main__":
    main()
