"""One-shot driver: spawn `claude` headless against C:/Projects/BlackjackXP
with a sharp prompt for a Flutter cross-platform (Windows desktop +
Android) Blackjack game. Mirrors `claude_runner.py` without needing
the gateway up.

Run:
    python scripts/spawn_blackjack_xp.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT = Path(r"C:/Projects/BlackjackXP")
TIMEOUT_S = 1800.0  # 30 min

PROMPT = """You are building a cross-platform Blackjack card game that runs on
Windows desktop AND Android, using Flutter.

Project directory: C:/Projects/BlackjackXP

Title: Build a cross-platform Blackjack game (Flutter — Android + Windows desktop)

Description:
A polished single-player Blackjack game (player vs dealer) implemented
in Flutter so the same codebase runs on Windows desktop and Android.
The game must be playable end-to-end with a clean UI: deal, hit, stand,
bust, dealer turn, win/lose/push result, play again.

Rules to model exactly:
  - Standard 52-card deck, shuffled. Number cards = face value, J/Q/K = 10,
    Ace = 1 or 11 (whichever keeps the hand under 21 when possible).
  - Goal: get as close to 21 as possible without going over.
  - Going over 21 = bust, immediate loss.
  - Dealer hits below 17 and on a SOFT 17. Stands on hard 17 and 18+.
  - A two-card 21 is a natural blackjack and beats a regular 21.

Implementation requirements:
  1. Run `flutter create .` in the project directory to scaffold a fresh
     Flutter app. Include platform support for `--platforms=windows,android`.
     Do NOT scaffold ios/macos/linux/web.
  2. Replace the boilerplate counter app with a Blackjack UI. Material
     widgets are fine. Show: dealer hand (one face-down before player
     stands), player hand, scores, buttons (Hit, Stand, Deal Again),
     result banner (win/lose/push/blackjack).
  3. Put the core game model (Card, Deck, Hand, Game state machine) in
     pure-Dart files under `lib/game/`. Keep it widget-free so it is
     unit-testable.
  4. Use a state-management approach that fits Flutter idioms — provider,
     ChangeNotifier, or Riverpod (your call). Do not pull a heavy
     framework just for one game.
  5. Write Dart unit tests under `test/` that cover:
       - Deck has 52 unique cards after shuffling
       - Hand total handles Ace 1 vs 11 correctly (e.g., A+7 = 18 soft,
         A+7+9 = 17 hard)
       - Dealer hits soft 17, stands on hard 17
       - Natural blackjack beats a 3-card 21
     Tests must pass via `flutter test`.
  6. Build for Windows desktop and confirm the build succeeds:
       flutter build windows --release
     Capture the exe path in the README.
  7. Try the Android build:
       flutter build apk --release
     If the build fails because the Android SDK path contains spaces,
     document the failure in README.md under a "Known Issues" section
     and continue — don't block on it. The Dart code itself must still
     be Android-compatible (no Windows-only APIs).
  8. Commit early and often with conventional-commit messages (feat:,
     test:, chore:). Do NOT push to any remote.

Acceptance criteria (ALL must pass):
  - `flutter test` passes with at least 6 tests covering the rules above
  - `flutter build windows --release` produces a runnable .exe
  - lib/game/ contains pure-Dart game model files
  - README.md exists and documents: how to run on Windows, how to run
    on Android, how to run tests, known issues
  - At least one commit per logical step

Files of interest (these globs should match):
  - pubspec.yaml
  - lib/main.dart
  - lib/game/**/*.dart
  - test/**/*_test.dart
  - README.md
  - windows/CMakeLists.txt

Rules:
  - Stay inside C:/Projects/BlackjackXP. Never edit files outside it.
  - Do not run `git push`. Local commits are fine.
  - Do not call paid APIs. No image-gen, no LLM calls.
  - Use the project's existing conventions (Flutter idioms, dart format,
    flutter analyze clean).
  - When done, ensure the acceptance criteria are met. Don't claim
    success otherwise.
"""


async def main() -> int:
    cli = shutil.which("claude")
    if cli is None:
        print("claude CLI not on PATH", file=sys.stderr)
        return 2
    PROJECT.mkdir(parents=True, exist_ok=True)
    args = [
        cli, "-p", PROMPT,
        "--output-format", "text",
        "--permission-mode", "bypassPermissions",
        "--add-dir", str(PROJECT),
    ]
    print(f"[driver] spawning claude in {PROJECT} (timeout={TIMEOUT_S}s)")
    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(PROJECT),
        env={**os.environ},
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        print(f"[driver] TIMEOUT after {TIMEOUT_S}s", file=sys.stderr)
        return 3
    dt = time.monotonic() - t0
    so = (stdout or b"").decode("utf-8", errors="replace")
    se = (stderr or b"").decode("utf-8", errors="replace")
    print(f"[driver] claude exited {proc.returncode} in {dt:.1f}s")
    print("=== STDOUT (tail) ===")
    print(so[-4000:])
    print("=== STDERR (tail) ===")
    print(se[-2000:])
    return proc.returncode if proc.returncode is not None else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
