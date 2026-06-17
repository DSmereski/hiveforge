"""One-shot driver: run the hive agent loop (planner-qwen) against
C:/Projects/BlackjackHive to build a Python Blackjack game from
scratch. Prints a transcript and final verdict.

Run:
    python scripts/spawn_blackjack_hive.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from gateway.crew_board.store import CrewBoardStore, Project
from gateway.crew_board import schema
from gateway.crew_board.hive_agent_loop import run_hive_agent_loop

import os as _os
PROJECT_PATH = Path(_os.environ.get("HIVE_BUILD_PROJECT_PATH", "./BlackjackHive"))
VAULT_DB = Path(_os.environ.get("HIVE_VAULT_PATH", "./vault")) / ".vault-writer" / "vault.db"

TASK_BODY = """Single-player Blackjack game in pure Python (>=3.10). The
player plays against the dealer in a CLI loop. Standard rules:

  - 52-card deck, shuffled. Numbers face value, J/Q/K = 10,
    Ace = 1 or 11 (whichever keeps total <= 21 when possible).
  - Player hits until they stand or bust (>21).
  - Dealer hits below 17 AND on a soft 17. Stands on hard 17 or 18+.
  - Two-card 21 = natural blackjack (beats a 3-card 21).
  - On bust the player loses immediately.

Layout (FLAT — package at root, no src/ dir):
  - blackjack/__init__.py — empty
  - blackjack/cards.py — Card, Suit, Rank, Deck (with .shuffle, .deal), build_deck()
  - blackjack/hand.py  — Hand with .add_card, .total, .is_soft, .is_blackjack
  - blackjack/game.py  — Game with dealer policy (hits soft 17)
  - blackjack/cli.py   — main() loop (input/output)
  - tests/__init__.py  — empty
  - tests/test_cards.py — deck has 52 unique cards
  - tests/test_hand.py  — A+7 = 18 soft, A+7+9 = 17 hard, 2-card 21 = blackjack
  - tests/test_game.py  — dealer hits soft 17, stands hard 17

IMPORTANT — tests import like this (no sys.path tricks):
    from blackjack.cards import Card, Deck, build_deck, Suit, Rank
    from blackjack.hand import Hand
    from blackjack.game import Game

pytest runs from project root and finds `blackjack/` because it's
at root. DO NOT create a src/ directory. DO NOT add sys.path.insert
to tests. DO NOT add __init__.py to the project root.

Make tests pass via `python -m pytest -q`. Commit when green.

REGRESSION RULE: when fixing a failing test, the count of PASSING
tests must NOT decrease. If you make a change and more tests fail
than before, immediately read_file the previous version and undo.

PYTHON GOTCHAS TO AVOID:
  - DO NOT give two enum members the same value. Python turns
    duplicates into aliases, so a `Rank` enum like:
        TEN = 10
        JACK = 10  # ← becomes an ALIAS of TEN; not a distinct member
        QUEEN = 10
        KING = 10
    iterates as only 10 unique members, giving a 40-card deck.
    Instead use distinct sentinel values and a separate value map:
        class Rank(Enum):
            TWO=2; THREE=3; ... TEN=10; JACK=11; QUEEN=12; KING=13; ACE=14
        BLACKJACK_VALUE = {Rank.JACK:10, Rank.QUEEN:10, Rank.KING:10,
                           Rank.ACE:11, ...}
"""


async def main() -> int:
    import argparse, subprocess
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="Keep existing project + use last pytest output "
                    "as initial context (don't wipe)")
    ap.add_argument("--max-iters", type=int, default=80)
    args_cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if not args_cli.resume:
        # Wipe + re-init
        import shutil
        if PROJECT_PATH.exists():
            shutil.rmtree(PROJECT_PATH, ignore_errors=True)
    PROJECT_PATH.mkdir(parents=True, exist_ok=True)
    if not (PROJECT_PATH / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=PROJECT_PATH, check=True)

    store = CrewBoardStore(VAULT_DB)
    proj = Project(
        slug="blackjackhive",
        path=str(PROJECT_PATH).replace("\\", "/"),
        name="BlackjackHive",
        enabled=True, push_allowed=False,
        test_cmd="python -m pytest -q",
    )
    store.upsert_project(proj)
    task = store.create_task(
        project_slug="blackjackhive",
        title="Build Python Blackjack (hive)",
        body=TASK_BODY,
        created_by="owner",
        acceptance_criteria=[
            {"text": "python -m pytest -q passes", "checked": False},
            {"text": "blackjack/ contains cards.py, hand.py, game.py, cli.py"},
            {"text": "tests/ has at least 3 test files"},
        ],
        files_of_interest=[
            "blackjack/__init__.py",
            "blackjack/cards.py",
            "blackjack/hand.py",
            "blackjack/game.py",
            "blackjack/cli.py",
            "tests/__init__.py",
            "tests/test_cards.py",
            "tests/test_hand.py",
            "tests/test_game.py",
        ],
        tags=["hive", "blackjack", "python"],
    )
    store.assign_task(task.slug, "hive", actor="owner")
    store.move_task(task.slug, schema.STATUS_READY, actor="owner",
                    detail="ready for hive loop")
    store.move_task(task.slug, schema.STATUS_IN_PROGRESS, actor="hive",
                    detail="hive loop start")
    task = store.get_task(task.slug)

    print(f"[driver] task {task.slug} in_progress; project at {PROJECT_PATH}")
    # Write transcript OUTSIDE the project dir so the model can't read
    # it back via list_dir/read_file and confuse itself.
    transcript_path = Path(_os.environ.get("HIVE_TRANSCRIPT_DIR", "./transcripts")) / "blackjack_hive_transcript.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: include latest pytest failure in seed so the model
    # picks up where the last attempt left off instead of starting fresh.
    initial_observation: str | None = None
    if args_cli.resume:
        try:
            r = subprocess.run(
                ["python", "-m", "pytest", "-q", "--tb=short"],
                cwd=str(PROJECT_PATH), capture_output=True, text=True,
                timeout=120,
            )
            initial_observation = (
                f"RESUMING — last `python -m pytest -q` returned exit "
                f"code {r.returncode}. Fix what's failing.\n\n"
                f"--- stdout ---\n{(r.stdout or '')[-1500:]}\n"
                f"--- stderr ---\n{(r.stderr or '')[-500:]}"
            )
            print(f"[driver] resume seed: pytest rc={r.returncode}")
        except Exception as e:  # noqa: BLE001
            print(f"[driver] resume seed failed: {e}")

    t0 = time.monotonic()
    result = await run_hive_agent_loop(
        store, task, max_iters=args_cli.max_iters,
        transcript_path=transcript_path,
        initial_observation=initial_observation,
    )
    dt = time.monotonic() - t0
    print(f"[driver] hive loop ok={result.ok} turns={result.turns} dt={dt:.1f}s")
    print(f"[driver] reason: {result.reason or '(none)'}")
    print(f"[driver] summary: {result.summary}")
    print(f"[driver] transcript: {transcript_path}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
