"""End-to-end Crew Board eval: create the Blackjack project, file a
task, assign Claude Code, wait for the dispatcher to land it in
Review, owner-approve, archive.

Run repeatedly with --attempt 1..5; each pass tightens the prompt and
acceptance criteria based on what the previous attempt missed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PROJECT_NAME = "Blackjack"
PROJECT_PATH = Path("C:/Projects") / PROJECT_NAME


def _task_for_attempt(attempt: int) -> dict:
    """Each attempt sharpens the body + criteria so the agent has
    progressively less room to misread."""
    bases = {
        1: dict(
            body=(
                "Build a playable single-player Blackjack game in Python "
                "for the project at C:/Projects/Blackjack. Standard rules: "
                "21 wins, bust on >21, dealer hits on <17. Provide a CLI "
                "you can run with `python -m blackjack` or `python blackjack.py`."
            ),
            criteria=[
                "File blackjack.py (or blackjack/__main__.py) exists",
                "Running the game from the project root completes without crashing",
                "A pytest test file exists and covers card-deal + bust logic",
            ],
        ),
        2: dict(
            body=(
                "Build a single-player Blackjack game in Python. Project: "
                "C:/Projects/Blackjack. Implement Card, Deck, Hand classes. "
                "Dealer must hit on totals < 17 (soft 17 hits too). Player "
                "can hit/stand each turn via a CLI prompt. Print clear "
                "win/lose/push messages. Add `requirements.txt` (empty if "
                "no deps), `README.md`, and pytest tests."
            ),
            criteria=[
                "src/blackjack/__init__.py and __main__.py exist",
                "tests/test_blackjack.py exists with at least 5 test cases",
                "pytest passes from the project root",
                "README.md explains how to run the game",
                "Game runs end-to-end without an unhandled exception",
            ],
        ),
        3: dict(
            body=(
                "Single-player CLI Blackjack in Python at C:/Projects/Blackjack. "
                "Strict deliverables: (1) blackjack package with Card/Deck/Hand/Game; "
                "(2) deterministic random seed support for tests; (3) hit/stand "
                "loop with input(); (4) dealer auto-plays; (5) pytest tests "
                "for hand-total, bust, dealer logic, and a full game with a "
                "fixed seed. Project layout: src/blackjack/, tests/, README.md, "
                "pyproject.toml (test cmd = pytest -q)."
            ),
            criteria=[
                "pyproject.toml exists and declares [build-system] + project name 'blackjack'",
                "src/blackjack/game.py defines a Game class with play() method",
                "tests/test_blackjack.py has >= 8 test cases",
                "pytest -q exits with code 0",
                "README.md documents install + run + test",
                "No file is empty",
            ],
        ),
        4: dict(
            body=(
                "Attempt 4 — production-grade. Single-player CLI Blackjack "
                "at C:/Projects/Blackjack. Architecture: src/blackjack/ "
                "package with separate modules card.py, deck.py, hand.py, "
                "game.py, cli.py. Type hints throughout. dataclass for "
                "Card. Hand exposes .total() handling Aces (1 vs 11). "
                "Game injects a random.Random instance for testability. "
                "CLI in cli.py provides a `main()` entry point. Tests must "
                "cover: ace soft/hard, bust threshold, dealer-stops-on-17, "
                "natural blackjack pays out, push on tie. pyproject.toml "
                "declares pytest as a dev dep."
            ),
            criteria=[
                "src/blackjack/{card,deck,hand,game,cli}.py all exist",
                "Hand.total() returns the BEST valid total accounting for aces",
                "Game accepts a `rng: random.Random` parameter",
                "tests cover ace soft/hard, bust, dealer-17, natural, push",
                "pytest -q exits with code 0",
                "All public functions have type hints",
                "No TODO or FIXME left in the source",
            ],
        ),
        5: dict(
            body=(
                "Final attempt — ship a clean Blackjack game at "
                "C:/Projects/Blackjack. Same architecture as attempt 4 plus: "
                "(a) a runner shim `blackjack.py` at project root that "
                "imports from src and calls cli.main(); (b) a Makefile or "
                "tasks.py with `test` and `run` targets; (c) acceptance "
                "test that plays a full game via a scripted input sequence "
                "and asserts the game terminated. Stick to the standard "
                "library — no third-party runtime deps."
            ),
            criteria=[
                "blackjack.py at project root runs the game when invoked",
                "src/blackjack/ package layout from attempt 4 is preserved",
                "tests/test_full_game.py runs an end-to-end game with scripted input",
                "Total test count >= 12",
                "pytest -q exits with code 0",
                "No imports outside the standard library in runtime code",
                "README documents make/tasks commands",
            ],
        ),
    }
    return bases[attempt]


async def wait_for_status(
    client: httpx.AsyncClient, slug: str,
    *, target: set[str], timeout_s: float = 1800.0,
) -> str | None:
    """Poll /board/state every 5s until the task hits `target` or timeout."""
    t0 = time.time()
    last_status = None
    while time.time() - t0 < timeout_s:
        r = await client.get("/board/state")
        if r.status_code != 200:
            await asyncio.sleep(5)
            continue
        s = r.json()
        task = next((t for t in s["tasks"] if t["slug"] == slug), None)
        if task is None:
            await asyncio.sleep(5)
            continue
        if task["status"] != last_status:
            print(
                f"  status={task['status']:12s} "
                f"assignee={task['assignee']:11s} "
                f"attempts={task['attempt_count']} "
                f"(+{int(time.time()-t0)}s)"
            )
            last_status = task["status"]
        if task["status"] in target:
            return task["status"]
        await asyncio.sleep(5)
    return None


async def run_attempt(attempt: int, *, host: str) -> bool:
    print(f"\n========== ATTEMPT {attempt} / 5 ==========\n")
    # Step 0: clear any prior Blackjack project on disk so each attempt
    # starts fresh.
    if PROJECT_PATH.exists():
        print(f"  cleanup: removing {PROJECT_PATH}")
        try:
            shutil.rmtree(PROJECT_PATH, ignore_errors=False)
        except OSError as e:
            print(f"  cleanup failed: {e}")

    async with httpx.AsyncClient(base_url=host, timeout=30.0) as c:
        # Step 1: ensure project exists (mkdir + git init + register).
        print("  creating project on board…")
        r = await c.post("/board/projects/create", json={
            "name": PROJECT_NAME, "path": str(PROJECT_PATH).replace("\\","/"),
        })
        if r.status_code != 200:
            print(f"  project create failed: {r.status_code} {r.text}")
            return False
        project_slug = r.json()["slug"]
        # Set test cmd so verifier runs pytest.
        # (project_scanner picks pytest because we'll have pyproject.toml.)
        # Step 2: create the task (owner-created → lands in backlog).
        spec = _task_for_attempt(attempt)
        r = await c.post("/board/tasks", json={
            "title": f"Build Blackjack game (attempt {attempt})",
            "project_slug": project_slug,
            "body": spec["body"],
            "priority": "high",
            "acceptance_criteria": [
                {"text": text, "checked": False} for text in spec["criteria"]
            ],
            "files_of_interest": [
                "blackjack.py", "src/blackjack/**/*.py",
                "tests/**/*.py", "README.md",
            ],
        })
        if r.status_code != 200:
            print(f"  task create failed: {r.status_code} {r.text}")
            return False
        task = r.json()
        slug = task["slug"]
        print(f"  task created: {slug}")
        # Step 3: assign to claude-code (hive can't make files).
        await c.post(f"/board/tasks/{slug}/assign", json={"assignee": "claude-code"})
        # Step 4: backlog -> ready (dispatcher picks up).
        await c.post(f"/board/tasks/{slug}/move", json={"status": "ready"})
        print("  task ready; waiting for dispatcher…")
        status = await wait_for_status(
            c, slug, target={"review", "done"}, timeout_s=1800.0,
        )
        if status is None:
            print(f"  timeout after 30 min; final state unknown")
            return False
        print(f"  task reached {status}")
        # Step 5: owner verifies by listing real files + running pytest.
        ok, reasons = _owner_verify(PROJECT_PATH, spec["criteria"])
        for r_msg in reasons:
            print(f"    {r_msg}")
        if not ok:
            print(f"  owner verdict: FAIL")
            return False
        # Step 6: mark criteria checked + move to done.
        r = await c.get("/board/state")
        task = next(t for t in r.json()["tasks"] if t["slug"] == slug)
        new_crit = [{**c, "checked": True} for c in task["acceptance_criteria"]]
        await c.post(
            f"/board/tasks/{slug}/criteria",
            json={"acceptance_criteria": new_crit},
        )
        if status == "review":
            await c.post(f"/board/tasks/{slug}/move", json={"status": "done"})
        print(f"  owner verdict: PASS, task moved to done")
        return True


def _owner_verify(project_path: Path, criteria: list[str]) -> tuple[bool, list[str]]:
    """Concrete checks the owner would do: files exist, pytest passes."""
    reasons: list[str] = []
    ok = True
    if not project_path.is_dir():
        return False, [f"project dir {project_path} does not exist"]
    # File existence sample.
    must_have_any_of = [
        ["blackjack.py", "src/blackjack/__main__.py", "src/blackjack/game.py"],
        ["tests/test_blackjack.py", "tests/test_full_game.py", "test_blackjack.py"],
        ["README.md"],
    ]
    for group in must_have_any_of:
        if not any((project_path / f).exists() for f in group):
            ok = False
            reasons.append(f"missing any of {group}")
    # Pytest run.
    try:
        proc = subprocess.run(
            ["pytest", "-q"], cwd=project_path,
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            ok = False
            reasons.append(f"pytest failed (exit {proc.returncode}). stderr tail: {(proc.stderr or '')[-400:]!r}")
        else:
            reasons.append(f"pytest passed: {proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else 'ok'}")
    except (subprocess.SubprocessError, OSError) as e:
        ok = False
        reasons.append(f"pytest could not be spawned: {e}")
    # Acceptance criteria — informational only (don't auto-tick).
    reasons.append(f"acceptance criteria count: {len(criteria)}")
    return ok, reasons


async def _run() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://127.0.0.1:8766")
    p.add_argument("--attempts", type=int, default=5)
    p.add_argument("--start-attempt", type=int, default=1)
    args = p.parse_args()
    results: list[tuple[int, bool]] = []
    for attempt in range(args.start_attempt, args.attempts + 1):
        ok = await run_attempt(attempt, host=args.host)
        results.append((attempt, ok))
        if ok:
            print(f"\nAttempt {attempt} succeeded — stopping.")
            break
    print("\n=== Summary ===")
    for attempt, ok in results:
        print(f"  attempt {attempt}: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    asyncio.run(_run())
