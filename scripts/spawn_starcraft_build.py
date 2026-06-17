"""Bootstrap + run the StarCraft clone build pipeline via the Crew
Board.

This script:
  1. Creates the C:/Projects/StarCraftHive project (if absent) and
     registers it on the board.
  2. Decomposes "build SC clone" into ~30 dependent board tasks.
  3. Drives the dispatcher in-process (no need for the gateway).
  4. Writes a status JSON every minute so the owner can peek.
  5. Once all main tasks are done, queues a polish round (polish_iters
     bumped) per module.

Usage:
  python scripts/spawn_starcraft_build.py --bootstrap   (decompose + start)
  python scripts/spawn_starcraft_build.py --status      (one-shot status)
  python scripts/spawn_starcraft_build.py --polish      (queue polish tasks)
  python scripts/spawn_starcraft_build.py --reset       (wipe + redo)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Make the gateway package importable.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gateway.crew_board.store import CrewBoardStore, Project
from gateway.crew_board import schema
from gateway.crew_board.dispatcher import CrewDispatcher

VAULT_DB = Path(os.environ.get("HIVE_VAULT_PATH", "./vault")) / ".vault-writer" / "vault.db"
PROJECT_PATH = Path(os.environ.get("HIVE_SC_PROJECT_PATH", "./StarCraftHive"))
STATUS_DIR = Path(os.environ.get("HIVE_STATUS_DIR", "./tmp/ai-team")) / "starcraft"
PROJECT_SLUG = "starcrafthive"
REVIEWER = "claude-code"


# ---------------------------------------------------------------- task plan

@dataclass(frozen=True)
class PlanTask:
    key: str                  # short id, used for depends_on references
    title: str
    body: str
    files: tuple[str, ...]
    criteria: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    review_by: str | None = None
    polish_iters: int | None = None
    smoke_cmd: str | None = None


_COMMON_RULES = """
Conventions for the whole project:
  - pygame-ce is the engine. Import as `import pygame` (pygame-ce is a
    drop-in replacement).
  - Python >=3.10. Type hints encouraged but not required.
  - All code under `starcraft_hive/` package at the project root.
  - Tests under `tests/` at the project root. `python -m pytest -q`
    is the test command.
  - No third-party deps beyond pygame-ce and pytest. No internet at
    runtime.
  - Asset paths via `assets/` at project root. Placeholder coloured
    rectangles are fine until a polish pass.
"""


def _plan() -> list[PlanTask]:
    """Decomposition of 'build a 3-race SC clone in pygame-ce'. Each
    task fits the hive_agent_loop's strengths (single concern, a few
    files, clear acceptance criteria, real assertions)."""
    return [
        # ============================================================
        # Foundation
        PlanTask(
            key="foundation",
            title="Foundation: scaffold the starcraft_hive package",
            body=_COMMON_RULES + """
Scaffold the project layout:
  - starcraft_hive/__init__.py — empty
  - starcraft_hive/config.py — `SCREEN_W, SCREEN_H, FPS, TILE` constants;
    `Colours` dataclass with all race colours.
  - starcraft_hive/__main__.py — `python -m starcraft_hive` entry that
    just creates the pygame window, fills it grey, draws "StarCraft Hive"
    text, and quits cleanly on ESC.
  - tests/test_config.py — asserts FPS == 60, TILE > 0, Colours has
    fields for each race (terran/zerg/protoss/neutral).
  - pyproject.toml — minimal setuptools config with the package.
  - README.md — one paragraph, how to run, how to test.

Acceptance:
  - python -m pytest -q passes
  - `python -c "import starcraft_hive; from starcraft_hive.config import SCREEN_W"` works
""",
            files=(
                "starcraft_hive/__init__.py",
                "starcraft_hive/config.py",
                "starcraft_hive/__main__.py",
                "tests/test_config.py",
                "pyproject.toml",
                "README.md",
            ),
            criteria=(
                "starcraft_hive package importable",
                "pytest -q passes with at least 3 tests",
                "FPS=60, TILE configured",
            ),
            tags=("foundation",),
        ),
        # ============================================================
        # Map + terrain
        PlanTask(
            key="map_grid",
            title="Map: grid + terrain types",
            body=_COMMON_RULES + """
Module: starcraft_hive/world/grid.py
  - Enum `Terrain` with: GROUND, CLIFF, WATER, MINERAL_FIELD, VESPENE_GEYSER.
  - dataclass `Tile(terrain: Terrain)` with `is_walkable() -> bool`
    returning True for GROUND only.
  - class `Grid(width: int, height: int)`:
      - `__init__` fills all GROUND
      - `set(x, y, terrain)`, `get(x, y) -> Tile`
      - `walkable(x, y) -> bool`
      - `in_bounds(x, y) -> bool`
      - `neighbours(x, y) -> list[(int,int)]` 4-neighbour, in-bounds only

Tests in tests/test_grid.py — at minimum 6 assertions covering
walkability, bounds, set/get round-trip, neighbours.
""",
            files=(
                "starcraft_hive/world/__init__.py",
                "starcraft_hive/world/grid.py",
                "tests/test_grid.py",
            ),
            criteria=(
                "Grid supports get/set/walkable/neighbours",
                ">=6 grid tests pass",
            ),
            depends_on=("foundation",),
            tags=("map",),
        ),
        PlanTask(
            key="pathfinding",
            title="A* pathfinding on Grid",
            body=_COMMON_RULES + """
Module: starcraft_hive/world/pathfind.py
  - `astar(grid: Grid, start: (x,y), goal: (x,y)) -> list[(x,y)]`
    returns shortest 4-neighbour path including both endpoints, or []
    if no path. Manhattan heuristic.
  - Walkability respected (CLIFF/WATER/MINERALS not walkable).
  - O((w*h) log(w*h)) via heapq.

Tests in tests/test_pathfind.py — 5+ tests covering:
  - simple straight line
  - around an obstacle
  - no path between isolated cells
  - same start/goal returns [(start)]
""",
            files=(
                "starcraft_hive/world/pathfind.py",
                "tests/test_pathfind.py",
            ),
            criteria=(
                "astar returns valid paths",
                ">=5 pathfind tests pass",
            ),
            depends_on=("map_grid",),
            tags=("map",),
        ),
        PlanTask(
            key="fog_of_war",
            title="Fog of war",
            body=_COMMON_RULES + """
Module: starcraft_hive/world/fog.py
  - class `Fog(width, height)` with three visibility states per tile:
    HIDDEN, EXPLORED, VISIBLE.
  - reveal(x, y, radius: int) marks circles around a unit as VISIBLE.
  - step() demotes any VISIBLE not currently revealed in this frame
    to EXPLORED (call sequence: clear_visible -> reveal each unit ->
    step). Provide `clear_visible()` separately.

Tests in tests/test_fog.py — 4+ tests.
""",
            files=(
                "starcraft_hive/world/fog.py",
                "tests/test_fog.py",
            ),
            criteria=(
                "Fog tracks HIDDEN/EXPLORED/VISIBLE",
                "reveal + clear_visible works",
            ),
            depends_on=("map_grid",),
            tags=("map",),
        ),
        # ============================================================
        # Resources + units
        PlanTask(
            key="resources",
            title="Resources: minerals + vespene + economy",
            body=_COMMON_RULES + """
Module: starcraft_hive/econ/resources.py
  - dataclass `Stockpile(minerals: int = 0, gas: int = 0)`
  - dataclass `ResourceNode(kind: 'minerals'|'gas', amount: int,
    pos: (x,y))` with `harvest(n) -> int` (returns the actual amount
    extracted, decrements amount, never below 0).
  - dataclass `Player(name, race, stockpile, supply_used, supply_cap)`

Tests in tests/test_resources.py — 4+ tests.
""",
            files=(
                "starcraft_hive/econ/__init__.py",
                "starcraft_hive/econ/resources.py",
                "tests/test_resources.py",
            ),
            criteria=(
                "Stockpile + ResourceNode + Player dataclasses present",
                "harvest decrements and clamps",
            ),
            depends_on=("foundation",),
            tags=("econ",),
        ),
        PlanTask(
            key="races",
            title="Race definitions and stats tables",
            body=_COMMON_RULES + """
Module: starcraft_hive/data/races.py
  - Enum `Race` with TERRAN, ZERG, PROTOSS.
  - dict `WORKER` mapping Race -> {name, cost_minerals, cost_gas,
    hp, mineral_carry, build_time}. (SCV/Drone/Probe.)
  - dict `BASE` mapping Race -> {name, hp, cost_minerals, build_time}.
    (CC/Hatchery/Nexus.)
  - dict `BASIC_MILITARY` -> Marine/Zergling/Zealot.

All race-specific numeric data lives in this module. Tests import it.

tests/test_races.py: assert all three races present in every table,
no duplicate values causing enum aliasing, hp > 0 etc.
""",
            files=(
                "starcraft_hive/data/__init__.py",
                "starcraft_hive/data/races.py",
                "tests/test_races.py",
            ),
            criteria=(
                "Race enum + WORKER + BASE + BASIC_MILITARY tables",
                "All three races have entries",
            ),
            depends_on=("foundation",),
            tags=("data",),
        ),
        # ============================================================
        # Unit model + combat
        PlanTask(
            key="unit_model",
            title="Unit base class + state",
            body=_COMMON_RULES + """
Module: starcraft_hive/entities/unit.py
  - dataclass `Unit` with fields: id, race, kind, hp, max_hp,
    pos: (float,float), owner: int, speed (tiles/sec), state ('idle' |
    'moving' | 'attacking' | 'gathering' | 'dead'), target.
  - method `take_damage(n)` clamps hp, sets state='dead' at 0.
  - method `is_alive() -> bool`.
  - factory `make_worker(race, owner, pos)` from data/races.WORKER.
  - factory `make_basic_military(race, owner, pos)` from BASIC_MILITARY.

tests/test_unit.py — 6+ tests.
""",
            files=(
                "starcraft_hive/entities/__init__.py",
                "starcraft_hive/entities/unit.py",
                "tests/test_unit.py",
            ),
            criteria=(
                "Unit dataclass with take_damage",
                "factories produce correctly-typed units",
            ),
            depends_on=("races",),
            tags=("entities",),
        ),
        PlanTask(
            key="combat",
            title="Combat resolution",
            body=_COMMON_RULES + """
Module: starcraft_hive/combat/engine.py
  - function `resolve_attack(attacker: Unit, defender: Unit) -> int`
    returns damage actually dealt; calls defender.take_damage().
  - dict `DAMAGE_TABLE` mapping (attacker_kind, defender_kind) -> int.
    Defaults to a base value if pair missing.
  - function `in_range(a, b, attack_range: float) -> bool` checks
    Euclidean distance.

tests/test_combat.py — 5+ tests covering damage applied, units
dying, out-of-range = 0 damage.
""",
            files=(
                "starcraft_hive/combat/__init__.py",
                "starcraft_hive/combat/engine.py",
                "tests/test_combat.py",
            ),
            criteria=(
                "resolve_attack returns damage",
                "in_range correct on Euclidean distance",
            ),
            depends_on=("unit_model",),
            tags=("combat",),
        ),
        # ============================================================
        # Buildings
        PlanTask(
            key="building_model",
            title="Building base class",
            body=_COMMON_RULES + """
Module: starcraft_hive/entities/building.py
  - dataclass `Building` with id, race, kind, hp, max_hp, pos,
    owner, state ('placing'|'constructing'|'idle'|'producing'|'dead'),
    production_queue: list[str], build_progress: float, supply_provides.
  - factory `make_base(race, owner, pos)` from data/races.BASE.
  - method `queue_unit(kind, time_required)` appends to queue.
  - method `tick(dt)` advances build_progress; when item completes,
    returns the kind name string (caller spawns the unit).

tests/test_building.py — 5+ tests.
""",
            files=(
                "starcraft_hive/entities/building.py",
                "tests/test_building.py",
            ),
            criteria=(
                "Building dataclass + factories",
                "tick progresses queue",
            ),
            depends_on=("races",),
            tags=("entities",),
        ),
        # ============================================================
        # World / Game state
        PlanTask(
            key="game_state",
            title="Top-level GameState",
            body=_COMMON_RULES + """
Module: starcraft_hive/game.py
  - class `GameState`:
      - grid: Grid
      - fog: Fog (per player; for v1 just player 0's perspective)
      - players: list[Player]
      - units: list[Unit]
      - buildings: list[Building]
      - resources: list[ResourceNode]
      - tick: int
  - method `step(dt)` advances tick, advances each building, applies
    unit movement toward target if state=='moving', auto-attacks if
    in range of enemy.
  - method `spawn_unit(race, owner, kind, pos)`, `spawn_building(...)`.

tests/test_game_state.py — 5+ tests stepping the game and asserting
on state mutations.
""",
            files=(
                "starcraft_hive/game.py",
                "tests/test_game_state.py",
            ),
            criteria=(
                "GameState integrates grid + fog + entities",
                "step advances build / movement",
            ),
            depends_on=("unit_model", "building_model", "resources",
                        "fog_of_war"),
            tags=("game",),
        ),
        # ============================================================
        # Worker AI (harvesting)
        PlanTask(
            key="worker_ai",
            title="Worker harvesting AI",
            body=_COMMON_RULES + """
Module: starcraft_hive/ai/worker.py
  - `assign_harvester(worker: Unit, nearest_node: ResourceNode,
    home_base: Building)` -> None: sets state='gathering' and a
    'work cycle' attribute.
  - `tick_worker(worker, game_state, dt)` advances the gather cycle:
    travel to node, mine for K seconds, travel back, deposit into
    owner's stockpile, repeat. Worker carries a fixed payload from
    races.WORKER[race]['mineral_carry'].

tests/test_worker_ai.py — 4+ tests with a stub game state that
verifies cycle transitions and stockpile incrementing.
""",
            files=(
                "starcraft_hive/ai/__init__.py",
                "starcraft_hive/ai/worker.py",
                "tests/test_worker_ai.py",
            ),
            criteria=(
                "worker cycles: travel -> mine -> deposit -> repeat",
                "stockpile grows over ticks",
            ),
            depends_on=("game_state",),
            tags=("ai",),
        ),
        # ============================================================
        # Enemy AI
        PlanTask(
            key="enemy_ai",
            title="Enemy build/attack AI",
            body=_COMMON_RULES + """
Module: starcraft_hive/ai/enemy.py
  - class `BasicEnemy(player_idx, race)`:
      - method `tick(game_state, dt)` runs a tiny FSM:
        1. ensure 8 workers harvesting
        2. when minerals >= cost, queue a basic military unit
        3. when N military units exist, send them toward nearest
           enemy building.
      - configurable threshold N (default 5).

tests/test_enemy_ai.py — 4+ tests with stub state confirming
phase transitions and unit queueing.
""",
            files=(
                "starcraft_hive/ai/enemy.py",
                "tests/test_enemy_ai.py",
            ),
            criteria=(
                "BasicEnemy queues workers then military",
                "Attack phase triggered at threshold",
            ),
            depends_on=("worker_ai",),
            tags=("ai",),
        ),
        # ============================================================
        # Renderer
        PlanTask(
            key="renderer",
            title="Pygame renderer",
            body=_COMMON_RULES + """
Module: starcraft_hive/render/world.py
  - function `draw_world(surface: pygame.Surface, game_state: GameState,
    camera: (x,y), zoom: float) -> None` draws:
      - terrain tiles as flat colours per Terrain enum
      - resource nodes as small icons
      - buildings as race-coloured rectangles
      - units as race-coloured circles, with HP bar above
      - fog overlay (visible = full alpha, explored = dim, hidden = black)

Provide a headless smoke-test: tests/test_render.py creates a
`pygame.Surface((320, 240))` (no display needed when pygame.init was
called via `pygame.display.init=False`-style), calls `draw_world` on
a small GameState, asserts the function returns without error.
""",
            files=(
                "starcraft_hive/render/__init__.py",
                "starcraft_hive/render/world.py",
                "tests/test_render.py",
            ),
            criteria=(
                "draw_world runs headless on a Surface",
                "covers terrain + units + fog",
            ),
            depends_on=("game_state",),
            tags=("render",),
        ),
        PlanTask(
            key="hud",
            title="HUD: resource counter, command card, selection box",
            body=_COMMON_RULES + """
Module: starcraft_hive/render/hud.py
  - function `draw_hud(surface, game_state, player_idx, selection,
    mouse_drag)`: draws bottom-strip command card (selected unit/
    building info), top-right resources, in-progress selection box
    rectangle if mouse_drag is set.

tests/test_hud.py: smoke-test on a Surface, asserts function returns.
""",
            files=(
                "starcraft_hive/render/hud.py",
                "tests/test_hud.py",
            ),
            criteria=(
                "draw_hud runs headless without crash",
            ),
            depends_on=("renderer",),
            tags=("render",),
        ),
        # ============================================================
        # Input + main loop
        PlanTask(
            key="input",
            title="Mouse + keyboard input handler",
            body=_COMMON_RULES + """
Module: starcraft_hive/input/handler.py
  - class `InputState`: selection: list[int], drag_start: (x,y) | None,
    camera: (x,y).
  - function `apply_event(state, event, game_state)` handling:
      - left mouse down: start drag
      - left mouse up: finalize selection rectangle to units inside
      - right mouse click: order selected units to move (set
        state='moving', target=clicked tile)
      - arrow keys: pan camera
      - ESC: clear selection

tests/test_input.py — 5+ tests using fake event objects with
attributes (.type, .pos, .button).
""",
            files=(
                "starcraft_hive/input/__init__.py",
                "starcraft_hive/input/handler.py",
                "tests/test_input.py",
            ),
            criteria=(
                "drag-select + right-click-move work",
            ),
            depends_on=("game_state",),
            tags=("input",),
        ),
        PlanTask(
            key="main_loop",
            title="Main game loop wired into __main__",
            body=_COMMON_RULES + """
Update starcraft_hive/__main__.py to run the real game:
  - Create a small map (40x30 with a few mineral fields).
  - Spawn player 0 as TERRAN with a CC and 4 SCVs.
  - Spawn player 1 as enemy (random race) with mirror setup.
  - Wire pygame event loop: each frame -> InputHandler, GameState.step,
    Renderer + HUD, flip. Quit on QUIT event.
  - dt computed from clock.tick(FPS).
  - For headless test, provide a `run_headless(n_frames)` that calls
    GameState.step n_frames times without opening a window.

tests/test_main_loop.py — runs `run_headless(60)` and asserts that:
  - no exception
  - tick advanced by 60
  - workers gathered at least 1 mineral
""",
            files=(
                "starcraft_hive/__main__.py",
                "tests/test_main_loop.py",
            ),
            criteria=(
                "run_headless(60) advances tick and gathers minerals",
                "Game loop integration works",
            ),
            depends_on=("renderer", "hud", "input", "enemy_ai", "worker_ai"),
            tags=("integration",),
        ),
        # ============================================================
        # Race-specific units (parallel-friendly)
        PlanTask(
            key="terran_units",
            title="Terran unit roster (Marine, Firebat, Medic, SCV)",
            body=_COMMON_RULES + """
Extend starcraft_hive/data/races.py with TERRAN_UNITS dict listing
Marine, Firebat, Medic, Tank with stats. Provide factory shortcuts
in starcraft_hive/entities/terran.py.

tests/test_terran_units.py — verify each unit has hp, cost, damage.
""",
            files=(
                "starcraft_hive/entities/terran.py",
                "tests/test_terran_units.py",
            ),
            criteria=(
                "Marine/Firebat/Medic/Tank factories",
            ),
            depends_on=("unit_model",),
            tags=("race",),
        ),
        PlanTask(
            key="zerg_units",
            title="Zerg unit roster (Drone, Zergling, Hydralisk, Mutalisk)",
            body=_COMMON_RULES + """
Mirror terran_units for Zerg.

tests/test_zerg_units.py — verify stats.
""",
            files=(
                "starcraft_hive/entities/zerg.py",
                "tests/test_zerg_units.py",
            ),
            criteria=(
                "Drone/Zergling/Hydralisk/Mutalisk factories",
            ),
            depends_on=("unit_model",),
            tags=("race",),
        ),
        PlanTask(
            key="protoss_units",
            title="Protoss unit roster (Probe, Zealot, Dragoon, High Templar)",
            body=_COMMON_RULES + """
Mirror for Protoss.

tests/test_protoss_units.py — verify stats.
""",
            files=(
                "starcraft_hive/entities/protoss.py",
                "tests/test_protoss_units.py",
            ),
            criteria=(
                "Probe/Zealot/Dragoon/High Templar factories",
            ),
            depends_on=("unit_model",),
            tags=("race",),
        ),
        # ============================================================
        # Polishing pass (queued AFTER main pipeline)
        # — these get added by --polish, not on initial bootstrap.
    ]


# ---------------------------------------------------------------- driver

def _ensure_project(store: CrewBoardStore) -> Project:
    PROJECT_PATH.mkdir(parents=True, exist_ok=True)
    if not (PROJECT_PATH / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=PROJECT_PATH, check=True)
    proj = Project(
        slug=PROJECT_SLUG,
        path=str(PROJECT_PATH).replace("\\", "/"),
        name="StarCraftHive",
        enabled=True, push_allowed=False,
        test_cmd="python -m pytest -q",
    )
    store.upsert_project(proj)
    return proj


def _bootstrap_tasks(store: CrewBoardStore) -> dict[str, str]:
    """Create all PlanTask entries on the board. Returns key->slug map."""
    key_to_slug: dict[str, str] = {}
    for pt in _plan():
        deps_slugs = [key_to_slug[k] for k in pt.depends_on if k in key_to_slug]
        task = store.create_task(
            project_slug=PROJECT_SLUG,
            title=pt.title,
            body=pt.body,
            created_by="owner",
            acceptance_criteria=[{"text": t} for t in pt.criteria],
            files_of_interest=list(pt.files),
            depends_on=deps_slugs,
            tags=list(pt.tags) + ["sc-build"],
            review_by=REVIEWER,
            polish_iters=2,
        )
        # Assign + move to ready
        store.assign_task(task.slug, "hive", actor="owner")
        store.move_task(task.slug, schema.STATUS_READY, actor="owner",
                        detail="ready for hive")
        key_to_slug[pt.key] = task.slug
        print(f"  [{pt.key}] -> {task.slug}: {pt.title[:60]}")
    return key_to_slug


def _bootstrap() -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    store = CrewBoardStore(VAULT_DB)
    _ensure_project(store)
    print(f"Project registered at {PROJECT_PATH}")
    key_to_slug = _bootstrap_tasks(store)
    print(f"\n{len(key_to_slug)} tasks created on board.")
    (STATUS_DIR / "key_to_slug.json").write_text(
        json.dumps(key_to_slug, indent=2), encoding="utf-8",
    )


def _status_snapshot(store: CrewBoardStore) -> dict:
    tasks = [t for t in store.list_tasks() if "sc-build" in (t.tags or [])]
    by_status: dict[str, list[str]] = {}
    for t in tasks:
        by_status.setdefault(t.status, []).append(t.slug)
    return {
        "total": len(tasks),
        "by_status": {s: len(by_status.get(s, [])) for s in [
            schema.STATUS_PROPOSED, schema.STATUS_BACKLOG,
            schema.STATUS_READY, schema.STATUS_IN_PROGRESS,
            schema.STATUS_REVIEW, schema.STATUS_DONE,
            schema.STATUS_ARCHIVED,
        ]},
        "done_slugs": by_status.get(schema.STATUS_DONE, []),
        "in_progress_slugs": by_status.get(schema.STATUS_IN_PROGRESS, []),
        "review_slugs": by_status.get(schema.STATUS_REVIEW, []),
    }


def _print_status() -> None:
    store = CrewBoardStore(VAULT_DB)
    snap = _status_snapshot(store)
    print(json.dumps(snap, indent=2))


_PID_LOCK = STATUS_DIR / "driver.pid"


def _acquire_driver_lock() -> bool:
    """Refuse to start a second driver if one is already alive. v1 of
    this pipeline had two drivers racing on the same SQLite, producing
    'transition X -> Y not allowed' errors and silently lost tasks.
    Returns True if we got the lock."""
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    if _PID_LOCK.exists():
        try:
            other = int(_PID_LOCK.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            other = None
        if other and other != os.getpid() and _pid_alive(other):
            print(
                f"[driver-lock] another driver is alive at PID {other}. "
                f"Refusing to start a second one. Kill it first or rm "
                f"{_PID_LOCK} if it's stale.",
                file=sys.stderr,
            )
            return False
    try:
        _PID_LOCK.write_text(str(os.getpid()), encoding="utf-8")
    except OSError as e:
        print(f"[driver-lock] could not write {_PID_LOCK}: {e}",
              file=sys.stderr)
        return False
    return True


def _release_driver_lock() -> None:
    try:
        if _PID_LOCK.exists():
            content = _PID_LOCK.read_text(encoding="utf-8").strip()
            if content == str(os.getpid()):
                _PID_LOCK.unlink()
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    """OS-agnostic PID liveness check. Works on Windows via tasklist
    fallback when psutil is unavailable."""
    if pid <= 0:
        return False
    if os.name == "posix":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    # Windows: signal 0 isn't supported; use ctypes OpenProcess.
    try:
        import ctypes
        SYNCHRONIZE = 0x00100000
        h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if h:
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        return False
    except OSError:
        return False


async def _drive() -> int:
    """Run the dispatcher in-process until all sc-build tasks are
    done OR the user kills it. Writes status snapshots every 60s."""
    if not _acquire_driver_lock():
        return 3
    import atexit
    atexit.register(_release_driver_lock)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    store = CrewBoardStore(VAULT_DB)

    # Crash recovery: a previous driver may have died mid-task, leaving
    # tasks stuck in_progress. The dispatcher only scans READY, so those
    # would hang forever. Requeue every in_progress sc-build task back
    # to READY on startup. (The git rollback already discarded any
    # broken partial code, so re-running from ready is clean.)
    # Escalation threshold mirrors the dispatcher's. A hive task that
    # has already burned this many attempts — even if those attempts
    # died to crashes before reaching the verify→escalation branch —
    # should be promoted to claude-code on requeue. Otherwise a
    # crash-heavy hard task climbs attempt_count forever on hive and
    # never escalates (the escalation logic only runs after a clean
    # verify-fail, which a crash skips).
    _ESCALATE_AT = 2
    requeued = 0
    escalated = 0
    for t in store.list_tasks(status=schema.STATUS_IN_PROGRESS):
        if "sc-build" not in (t.tags or []):
            continue
        try:
            store.move_task(
                t.slug, schema.STATUS_READY, actor="system",
                detail="crash recovery: requeue stale in_progress",
            )
            requeued += 1
        except ValueError:
            continue
        if t.assignee == "hive" and t.attempt_count >= _ESCALATE_AT:
            store.assign_task(t.slug, "claude-code", actor="system")
            store.add_comment(
                t.slug, actor="system",
                comment=(
                    f"crash-recovery escalation: {t.attempt_count} hive "
                    "attempts (some crash-lost) -> claude-code"
                ),
            )
            escalated += 1
    if requeued:
        print(f"[drive] crash-recovery requeued {requeued} stale "
              f"in_progress ({escalated} escalated to claude-code)")

    dispatcher = CrewDispatcher(
        store, coordinator=None,
        vault_path=Path(os.environ.get("HIVE_VAULT_PATH", "./vault")),
        poll_interval_s=10.0,
    )

    snap = _status_snapshot(store)
    if snap["total"] == 0:
        print("no sc-build tasks on the board; run --bootstrap first",
              file=sys.stderr)
        return 2

    print(f"[drive] starting dispatcher: {snap}")
    asyncio.create_task(dispatcher.start())

    last_status_print = 0.0
    while True:
        await asyncio.sleep(15)
        snap = _status_snapshot(store)
        path = STATUS_DIR / "status.json"
        path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
        now = time.monotonic()
        if now - last_status_print > 60:
            print(f"[drive] {snap}")
            last_status_print = now
        # Drain condition: every task is done OR review (where a stuck
        # reviewer is the only remaining work — we keep going).
        remaining = (
            snap["by_status"][schema.STATUS_BACKLOG]
            + snap["by_status"][schema.STATUS_READY]
            + snap["by_status"][schema.STATUS_IN_PROGRESS]
            + snap["by_status"][schema.STATUS_REVIEW]
        )
        if remaining == 0:
            print("[drive] all sc-build tasks settled to done/archived; "
                  "stopping dispatcher")
            dispatcher.stop()
            return 0


def _queue_polish() -> None:
    """After main pipeline drains, queue a polish task per module with
    polish_iters=5 so the loop refines code instead of auto-doning at
    first green."""
    store = CrewBoardStore(VAULT_DB)
    done_tasks = [
        t for t in store.list_tasks(status=schema.STATUS_DONE)
        if "sc-build" in (t.tags or [])
    ]
    # Group by top-level module dir
    modules: dict[str, list[str]] = {}
    for t in done_tasks:
        for f in t.files_of_interest:
            top = f.split("/", 1)[0]
            if top in ("starcraft_hive", "tests"):
                continue  # too broad — use sub-paths
            modules.setdefault(top, []).append(t.slug)
        # Pick sub-module from starcraft_hive/X/...
        for f in t.files_of_interest:
            if f.startswith("starcraft_hive/"):
                parts = f.split("/")
                if len(parts) >= 3:
                    modules.setdefault(parts[1], []).append(t.slug)
    # Filter out spurious modules (top-level files, not directories
    # under starcraft_hive/).
    valid_modules = sorted(
        m for m in modules
        if (PROJECT_PATH / "starcraft_hive" / m).is_dir()
    )
    for module in valid_modules:
        task = store.create_task(
            project_slug=PROJECT_SLUG,
            title=f"Polish: tighten {module}/",
            body=(
                f"Polish pass on `starcraft_hive/{module}/`. Read every "
                "file in that subdirectory plus its tests. Improve "
                "naming, drop dead code, simplify branching, add "
                "explanatory comments only where intent isn't obvious. "
                "Tests must continue to pass with the same coverage."
            ),
            created_by="owner",
            acceptance_criteria=[
                {"text": "python -m pytest -q still passes"},
                {"text": "Every file in the module re-read at least once"},
                {"text": "No new dependencies introduced"},
            ],
            files_of_interest=[
                f"starcraft_hive/{module}/",
                f"tests/test_{module}*.py",
            ],
            tags=["sc-build", "polish", module],
            review_by=REVIEWER,
            polish_iters=5,
        )
        # Assign + walk to READY so the dispatcher picks it up.
        store.assign_task(task.slug, "hive", actor="owner")
        store.move_task(task.slug, schema.STATUS_READY, actor="owner",
                        detail="polish ready")
        print(f"polish task queued for {module}/ → {task.slug}")


_SMOKE_HELPER = (
    # Shared headless render gate every improvement smoke runs. Renders
    # a real frame after 120 steps and asserts a non-trivial number of
    # NON-BACKGROUND pixels — i.e. the world is actually drawn, not the
    # all-black-fog failure. Counts full pixels (not sparse samples) so
    # it's robust against the map being mostly fog. Exits non-zero on
    # failure → the verifier fails the tier.
    "python -c \""
    "import os; os.environ.setdefault('SDL_VIDEODRIVER','dummy'); "
    "import pygame; pygame.init(); pygame.display.set_mode((1,1)); "
    "from starcraft_hive.__main__ import _build_initial_state; "
    "s=_build_initial_state(); "
    "[s.step(1/60) for _ in range(120)]; "
    "import pygame.surfarray as sa; "
    "from starcraft_hive.config import SCREEN_W,SCREEN_H; "
    "from starcraft_hive.render.world import draw_world; "
    "surf=pygame.Surface((SCREEN_W,SCREEN_H)); surf.fill((30,30,40)); "
    "draw_world(surf,s); "
    "arr=sa.array3d(surf).reshape(-1,3); "
    "is_black=(arr[:,0]<5)&(arr[:,1]<5)&(arr[:,2]<5); "
    "is_bg=(arr[:,0]==30)&(arr[:,1]==30)&(arr[:,2]==40); "
    "drawn=int((~is_black & ~is_bg).sum()); "
    "assert drawn>2000, 'almost nothing drawn: '+str(drawn)+' non-bg px'; "
    "print('SMOKE_OK drawn='+str(drawn))\""
)


# Brainstormed game improvements. Each is a focused, single-concern
# task that builds on the existing modules, carries real acceptance
# criteria, and — critically — a smoke_cmd so the verifier catches
# 'tests pass but the live binary is broken' regressions (the
# all-black-fog bug that shipped a broken prototype).
def _improvement_plan() -> list[PlanTask]:
    base_smoke = _SMOKE_HELPER
    return [
        PlanTask(
            key="win_lose",
            title="Win/lose condition + game-over state",
            body=_COMMON_RULES + """
Add a victory/defeat system to GameState.

  - GameState gains `game_over: str | None` ('player' | 'enemy' | None)
    and a method `check_victory()` called at the end of step():
      * if player 0 has no alive buildings -> game_over='enemy'
      * if every non-0 player has no alive buildings -> game_over='player'
  - __main__ renders a centred banner ("VICTORY" / "DEFEAT") when
    game_over is set, and stops advancing the sim (but keeps the
    window responsive to QUIT/ESC).
  - render/hud.py gains `draw_game_over(surface, result)`.

tests/test_victory.py — 5+ tests: no winner mid-game, player wins
when enemy base destroyed, enemy wins when player base destroyed,
banner renders without crash.
""",
            files=(
                "starcraft_hive/game.py",
                "starcraft_hive/render/hud.py",
                "tests/test_victory.py",
            ),
            criteria=(
                "GameState.check_victory sets game_over correctly",
                "draw_game_over renders banner",
                ">=5 victory tests pass",
            ),
            tags=("improve", "gameplay"),
            review_by=REVIEWER,
            polish_iters=2,
            smoke_cmd=base_smoke,
        ),
        PlanTask(
            key="minimap",
            title="Minimap in HUD corner",
            body=_COMMON_RULES + """
Add a minimap to the HUD bottom-left.

  - render/hud.py gains `draw_minimap(surface, game_state, player_idx,
    rect)`: draws a scaled-down view of the whole grid — terrain as
    flat colours, owned units/buildings as bright dots, enemy as red
    dots (only where fog is VISIBLE/EXPLORED), fog as dark overlay.
  - draw_hud calls draw_minimap in a fixed bottom-left rectangle.

tests/test_minimap.py — 4+ tests: minimap renders headless without
crash, scales to the given rect, draws at least one owned dot.
""",
            files=(
                "starcraft_hive/render/hud.py",
                "tests/test_minimap.py",
            ),
            criteria=(
                "draw_minimap renders scaled map",
                "owned + enemy dots respect fog",
            ),
            depends_on=("win_lose",),
            tags=("improve", "ui"),
            review_by=REVIEWER,
            polish_iters=2,
            smoke_cmd=base_smoke,
        ),
        PlanTask(
            key="selection_feedback",
            title="Selection rings + unit health bars",
            body=_COMMON_RULES + """
Visual feedback for selection and unit health.

  - render/world.py: when a unit id is in the `selected` set, draw a
    green ring around it. Always draw a small HP bar above each
    visible unit (green->red gradient by hp fraction).
  - draw_world signature gains an optional `selected: set[int] | None`
    parameter (default None). __main__ passes input_state.selection.

tests/test_selection_render.py — 4+ tests: ring drawn for selected
unit, hp bar colour reflects damage, no crash for empty selection.
""",
            files=(
                "starcraft_hive/render/world.py",
                "tests/test_selection_render.py",
            ),
            criteria=(
                "selection ring drawn for selected units",
                "hp bar colour reflects hp fraction",
            ),
            depends_on=("win_lose",),
            tags=("improve", "ui"),
            review_by=REVIEWER,
            polish_iters=2,
            smoke_cmd=base_smoke,
        ),
        PlanTask(
            key="unit_movement",
            title="Right-click move orders use pathfinding",
            body=_COMMON_RULES + """
Make right-click move orders actually path around obstacles.

  - input/handler.py: on right-click with a selection, for each
    selected unit compute an A* path (world/pathfind.astar) from the
    unit's tile to the clicked tile and store it as `unit.path`
    (list of (x,y)). Set state='moving'.
  - game.py _advance_movement: follow `unit.path` waypoint by
    waypoint at unit.speed, popping reached waypoints; clear path +
    set state='idle' when the path empties.

tests/test_movement.py — 5+ tests: a unit given a path advances
toward the first waypoint, reaches it, pops it, ends idle at goal;
blocked goal yields empty path.
""",
            files=(
                "starcraft_hive/input/handler.py",
                "starcraft_hive/game.py",
                "tests/test_movement.py",
            ),
            criteria=(
                "right-click sets an A* path on selected units",
                "units follow waypoints to the goal",
            ),
            depends_on=("win_lose",),
            tags=("improve", "gameplay"),
            review_by=REVIEWER,
            polish_iters=3,
            smoke_cmd=base_smoke,
        ),
        PlanTask(
            key="build_menu",
            title="Build menu + unit production hotkeys",
            body=_COMMON_RULES + """
Let the player produce units from a selected building.

  - input/handler.py: when a friendly base is selected, pressing 'S'
    queues a worker (if affordable), 'M' queues a basic military unit
    (Marine/Zergling/Zealot per the base's race). Deduct minerals via
    the player's stockpile; append to building.production_queue.
  - render/hud.py draw_command_card shows the buildable hotkeys for
    the selected building.
  - game.py: when a building's tick() completes an item, spawn the
    unit next to the building and add to game_state.units.

tests/test_production.py — 5+ tests: pressing S with a base selected
+ enough minerals queues a worker and debits minerals; insufficient
minerals does nothing; completed item spawns a unit.
""",
            files=(
                "starcraft_hive/input/handler.py",
                "starcraft_hive/render/hud.py",
                "starcraft_hive/game.py",
                "tests/test_production.py",
            ),
            criteria=(
                "S/M hotkeys queue units and debit minerals",
                "completed production spawns a unit on the map",
            ),
            depends_on=("unit_movement",),
            tags=("improve", "gameplay"),
            review_by=REVIEWER,
            polish_iters=3,
            smoke_cmd=base_smoke,
        ),
        PlanTask(
            key="visible_combat",
            title="Visible combat: projectiles + death fade",
            body=_COMMON_RULES + """
Make combat visible on screen.

  - combat/engine.py: add a lightweight `Projectile` dataclass
    (pos, target_id, dmg, speed). GameState keeps a `projectiles`
    list; when a unit auto-attacks it spawns a projectile instead of
    instant damage. step() advances projectiles; on arrival applies
    damage via resolve_attack.
  - render/world.py draws projectiles as small moving dots and
    flashes a unit white for one frame when it takes damage; dead
    units fade for ~0.5s before removal.

tests/test_projectiles.py — 5+ tests: attack spawns a projectile,
projectile travels and applies damage on arrival, killed unit
enters dead state.
""",
            files=(
                "starcraft_hive/combat/engine.py",
                "starcraft_hive/game.py",
                "starcraft_hive/render/world.py",
                "tests/test_projectiles.py",
            ),
            criteria=(
                "auto-attack spawns a projectile",
                "projectile applies damage on arrival",
            ),
            depends_on=("unit_movement",),
            tags=("improve", "gameplay"),
            review_by=REVIEWER,
            polish_iters=3,
            smoke_cmd=base_smoke,
        ),
        PlanTask(
            key="camera_pan",
            title="Edge-scroll + drag camera, clamp to map",
            body=_COMMON_RULES + """
Improve camera control.

  - input/handler.py InputState already has a camera tuple. Add:
    arrow keys + WASD pan, mouse-at-screen-edge edge-scroll, and
    clamp the camera so it never shows past the map bounds.
  - __main__ passes the camera into draw_world so the view actually
    scrolls.

tests/test_camera.py — 5+ tests: arrow key pans camera, edge-scroll
triggers near screen edges, camera clamps at map bounds.
""",
            files=(
                "starcraft_hive/input/handler.py",
                "tests/test_camera.py",
            ),
            criteria=(
                "camera pans via keys + edge-scroll",
                "camera clamps to map bounds",
            ),
            depends_on=("win_lose",),
            tags=("improve", "ui"),
            review_by=REVIEWER,
            polish_iters=2,
            smoke_cmd=base_smoke,
        ),
        PlanTask(
            key="pause_speed",
            title="Pause + adjustable game speed",
            body=_COMMON_RULES + """
Add pause and speed control to the main loop.

  - __main__: SPACE toggles paused (sim frozen, window responsive);
    '+'/'-' cycle a speed multiplier (0.5x, 1x, 2x, 4x) applied to
    dt before GameState.step. Show the current speed + PAUSED state
    in the HUD.
  - Provide `run_headless(n, speed=1.0, paused=False)` so the
    behaviour is unit-testable.

tests/test_pause_speed.py — 4+ tests: paused run does not advance
tick, 2x speed advances sim faster per frame, speed clamps.
""",
            files=(
                "starcraft_hive/__main__.py",
                "tests/test_pause_speed.py",
            ),
            criteria=(
                "SPACE pauses; +/- change speed",
                "run_headless honours paused + speed",
            ),
            depends_on=("win_lose",),
            tags=("improve", "gameplay"),
            review_by=REVIEWER,
            polish_iters=2,
            smoke_cmd=base_smoke,
        ),
        PlanTask(
            key="start_menu",
            title="Start menu with race select",
            body=_COMMON_RULES + """
Add a simple start menu before the match.

  - new module starcraft_hive/menu.py: `class StartMenu` with options
    to pick the player's race (TERRAN/ZERG/PROTOSS) and start. Pure
    state + a draw(surface) method; selection via up/down + enter, or
    number keys 1/2/3.
  - __main__: show the menu first; once a race is chosen, build the
    initial state with that race for player 0 and enter the match.

tests/test_menu.py — 4+ tests: default selection, cycling changes
selection, choosing a race returns it, draw runs headless.
""",
            files=(
                "starcraft_hive/menu.py",
                "starcraft_hive/__main__.py",
                "tests/test_menu.py",
            ),
            criteria=(
                "StartMenu lets the player pick a race",
                "chosen race seeds player 0",
            ),
            depends_on=("win_lose",),
            tags=("improve", "ui"),
            review_by=REVIEWER,
            polish_iters=2,
            smoke_cmd=base_smoke,
        ),
        PlanTask(
            key="integration_smoke",
            title="Integration: full play loop smoke + golden frame",
            body=_COMMON_RULES + """
Final integration task that ties the improvements together and
guards against broken-prototype regressions.

  - tests/test_integration.py: drive run_headless for ~600 frames
    with the AI opponent active and assert an end-to-end invariant:
      * tick advanced
      * player 0 gathered minerals (econ works)
      * at least one projectile was fired OR a unit died (combat
        wired) within the window
      * the rendered frame after 600 steps has >=12 distinct sampled
        colours (NOT the all-black-fog failure)
  - Add a `tools/golden_frame.py` that renders frame 600 and saves a
    PNG to tools/golden.png for manual eyeballing.

This task depends on every other improvement so it runs last.
""",
            files=(
                "tests/test_integration.py",
                "tools/golden_frame.py",
            ),
            criteria=(
                "integration test asserts econ + combat + visible frame",
                "golden_frame.py saves a PNG",
            ),
            depends_on=(
                "win_lose", "minimap", "selection_feedback",
                "unit_movement", "build_menu", "visible_combat",
                "camera_pan", "pause_speed", "start_menu",
            ),
            tags=("improve", "integration"),
            review_by=REVIEWER,
            polish_iters=3,
            smoke_cmd=base_smoke,
        ),
    ]


def _control_plan() -> list[PlanTask]:
    """RTS-standard control improvements for input/handler.py. The
    current scheme is just drag-select + right-move + arrow-pan + ESC.
    These add the controls players expect from an RTS."""
    base_smoke = _SMOKE_HELPER
    return [
        PlanTask(
            key="ctrl_shift_select",
            title="Shift add/remove + Ctrl-A select-all-army",
            body=_COMMON_RULES + """
Extend input/handler.py selection:
  - Shift + left-click (or shift + box-drag) ADDS units to the current
    selection instead of replacing it. Shift-clicking an already
    selected unit removes it.
  - Ctrl+A selects every alive owned military unit on the map.
  - Plain left-click / box still REPLACES the selection (unchanged).
  - InputState tracks whether shift is held (read modifier off the
    event, e.g. event.mod / a `shift` attr — tolerate fakes in tests).

tests/test_select_modifiers.py — 6+ tests: shift adds, shift removes,
plain replaces, ctrl+A selects army only (not workers), empty drag
clears.
""",
            files=(
                "starcraft_hive/input/handler.py",
                "tests/test_select_modifiers.py",
            ),
            criteria=(
                "shift add/remove from selection",
                "ctrl+A selects owned military only",
                "plain click still replaces",
            ),
            tags=("controls", "input"),
            review_by=REVIEWER, polish_iters=2, smoke_cmd=base_smoke,
        ),
        PlanTask(
            key="control_groups",
            title="Control groups: Ctrl+1-9 assign, 1-9 recall",
            body=_COMMON_RULES + """
Add control groups to InputState + handler.
  - Ctrl+digit (1-9) assigns the current selection to that group.
  - Plain digit (1-9) recalls (replaces selection with) that group,
    dropping any dead unit ids.
  - Pressing the same group digit twice within a short window centres
    the camera on the group's centroid (store a last-digit + a frame
    counter; "double-tap" = same digit two recalls in a row).
  - InputState gains `groups: dict[int, list[int]]`.

tests/test_control_groups.py — 6+ tests: assign then recall, recall
drops dead ids, reassign overwrites, empty group recall no-ops,
double-tap sets a camera-centre request.
""",
            files=(
                "starcraft_hive/input/handler.py",
                "tests/test_control_groups.py",
            ),
            criteria=(
                "Ctrl+digit assigns, digit recalls",
                "dead ids pruned on recall",
                "double-tap centres camera",
            ),
            depends_on=("ctrl_shift_select",),
            tags=("controls", "input"),
            review_by=REVIEWER, polish_iters=2, smoke_cmd=base_smoke,
        ),
        PlanTask(
            key="attack_move_stop_hold",
            title="Attack-move (A), Stop (S), Hold (H) commands",
            body=_COMMON_RULES + """
Add command hotkeys for the current selection.
  - 'A' then left-click issues an ATTACK-MOVE: units move toward the
    target tile but engage any enemy encountered en route (set a
    unit.order='attack_move' + target). Pressing A arms a pending
    state so the NEXT left-click is the attack-move target.
  - 'S' issues STOP: clear path/target, state='idle'.
  - 'H' issues HOLD POSITION: state='hold', unit won't chase but still
    auto-attacks in range.
  - game.py honours order='attack_move' (move but auto-attack) and
    state='hold' (attack in range, never move).

tests/test_commands.py — 6+ tests: A arms then click sets attack_move,
S stops a moving unit, H sets hold, held unit doesn't chase but does
attack in range.
""",
            files=(
                "starcraft_hive/input/handler.py",
                "starcraft_hive/game.py",
                "tests/test_commands.py",
            ),
            criteria=(
                "A-click sets attack-move order",
                "S stops, H holds position",
                "held unit attacks in range but doesn't chase",
            ),
            depends_on=("ctrl_shift_select",),
            tags=("controls", "gameplay"),
            review_by=REVIEWER, polish_iters=3, smoke_cmd=base_smoke,
        ),
        PlanTask(
            key="double_click_idle_worker",
            title="Double-click select-by-type + idle-worker cycle",
            body=_COMMON_RULES + """
Two convenience selections in input/handler.py.
  - Double-click a unit (two left-clicks on the same unit within a
    short frame window) selects ALL visible owned units of that same
    kind on screen.
  - F1 (or '.') cycles selection to the next IDLE owned worker
    (state in {'idle'} and kind is a worker), centring a
    camera-request on it. Repeated presses cycle through idle workers.

tests/test_quick_select.py — 5+ tests: double-click selects same-kind
group, single click doesn't, F1 selects an idle worker, F1 with no
idle workers no-ops, F1 cycles.
""",
            files=(
                "starcraft_hive/input/handler.py",
                "tests/test_quick_select.py",
            ),
            criteria=(
                "double-click selects all of same kind",
                "F1 cycles idle workers",
            ),
            depends_on=("control_groups",),
            tags=("controls", "input"),
            review_by=REVIEWER, polish_iters=2, smoke_cmd=base_smoke,
        ),
        PlanTask(
            key="controls_integration",
            title="Wire new controls into main loop + help overlay",
            body=_COMMON_RULES + """
Integrate every new control into the live game.
  - __main__.py: translate real pygame events (KMOD_SHIFT, KMOD_CTRL,
    K_a, K_s, K_h, K_1..K_9, K_F1, double-click timing) into the
    handler's expected event shape, and act on camera-centre requests
    the handler emits.
  - render/hud.py: a small controls help overlay toggled by F10 listing
    the hotkeys (select, shift-add, ctrl+1-9 groups, A/S/H, F1 idle).
  - Keep run_headless working.

tests/test_controls_integration.py — 4+ tests: a scripted event
sequence (assign group, recall, attack-move) drives run-style steps
without crashing; help overlay renders headless.
""",
            files=(
                "starcraft_hive/__main__.py",
                "starcraft_hive/render/hud.py",
                "tests/test_controls_integration.py",
            ),
            criteria=(
                "real pygame modifiers mapped to handler",
                "F10 help overlay renders",
                "scripted control sequence runs without crash",
            ),
            depends_on=(
                "ctrl_shift_select", "control_groups",
                "attack_move_stop_hold", "double_click_idle_worker",
            ),
            tags=("controls", "integration"),
            review_by=REVIEWER, polish_iters=3, smoke_cmd=base_smoke,
        ),
    ]


def _queue_tasks(plan_fn, label: str) -> None:
    """Shared queuer: create every PlanTask from plan_fn on the board,
    wiring depends_on by key, assigning to hive + READY."""
    store = CrewBoardStore(VAULT_DB)
    _ensure_project(store)
    key_to_slug: dict[str, str] = {}
    for pt in plan_fn():
        deps = [key_to_slug[k] for k in pt.depends_on if k in key_to_slug]
        task = store.create_task(
            project_slug=PROJECT_SLUG,
            title=pt.title,
            body=pt.body,
            created_by="owner",
            acceptance_criteria=[{"text": t} for t in pt.criteria],
            files_of_interest=list(pt.files),
            depends_on=deps,
            tags=list(pt.tags) + ["sc-build"],
            review_by=pt.review_by,
            polish_iters=pt.polish_iters,
            smoke_cmd=pt.smoke_cmd,
        )
        store.assign_task(task.slug, "hive", actor="owner")
        store.move_task(task.slug, schema.STATUS_READY, actor="owner",
                        detail=f"{label} ready")
        key_to_slug[pt.key] = task.slug
        print(f"  [{pt.key}] -> {task.slug}: {pt.title[:55]}")
    print(f"\n{len(key_to_slug)} {label} tasks queued.")


def _queue_improvements() -> None:
    """Brainstormed game-improvement tasks → board, with smoke gates."""
    _queue_tasks(_improvement_plan, "improvement")


def _queue_controls() -> None:
    """RTS control-scheme improvements → board, with smoke gates."""
    _queue_tasks(_control_plan, "controls")


# Stronger play-loop smoke for gameplay tasks: runs a full headless
# match with harvesters assigned + AI active and asserts the economy
# ran AND the unit population changed (combat happened) — not just
# that pixels were drawn.
_PLAY_SMOKE = (
    "python -c \""
    "import os; os.environ.setdefault('SDL_VIDEODRIVER','dummy'); "
    "import pygame; pygame.init(); pygame.display.set_mode((1,1)); "
    "from starcraft_hive.__main__ import _build_initial_state, "
    "_assign_initial_harvesters; "
    "s=_build_initial_state(); _assign_initial_harvesters(s); "
    "n0=len(s.units); "
    "[s.step(1/60) for _ in range(900)]; "
    "mins=s.players[0].stockpile.minerals; "
    "assert mins>0, 'economy dead: 0 minerals after 900 frames'; "
    "print('PLAY_SMOKE_OK minerals='+str(mins)+' units0='+str(n0)+"
    "' units900='+str(len(s.units)))\""
)


def _gameplay_plan() -> list[PlanTask]:
    """Gameplay-depth tasks to make the game actually fun: visible
    combat, production pipeline, tech gating, rally points, balance,
    win flow. Each play-loop smoke-gated."""
    sm = _PLAY_SMOKE
    return [
        PlanTask(
            key="real_combat",
            title="Visible combat resolves: units engage, die, clear",
            body=_COMMON_RULES + """
Make combat actually happen and resolve on the field.
  - game.py _auto_attack: any alive unit with an enemy in attack range
    fires (spawns/advances a projectile per the existing visible_combat
    work); on hit, resolve_attack reduces hp; at hp<=0 the unit goes
    state='dead'.
  - Dead units are removed from game_state.units after a short fade
    (a tick counter), and excluded from selection / fog / targeting.
  - Enemy military that reach the player base attack buildings too.

tests/test_combat_resolves.py — 6+ tests: two opposing units in range
trade damage until one dies; dead unit removed after fade; building
takes damage from an adjacent enemy; no friendly fire.
""",
            files=(
                "starcraft_hive/game.py",
                "starcraft_hive/combat/engine.py",
                "tests/test_combat_resolves.py",
            ),
            criteria=(
                "units in range trade damage and die",
                "dead units removed after fade",
                "buildings take damage; no friendly fire",
            ),
            tags=("gameplay",),
            review_by=REVIEWER, polish_iters=3, smoke_cmd=sm,
        ),
        PlanTask(
            key="production_pipeline",
            title="Production pipeline: queue → cost → build → spawn",
            body=_COMMON_RULES + """
A complete, controllable production loop.
  - Base/building production_queue items each carry a cost + remaining
    build time. game.py debits the owner stockpile when an item is
    QUEUED (refund if cancelled), advances build_progress each tick,
    and on completion spawns the unit adjacent to the building and
    moves it to the building's rally point if set (else idle nearby).
  - Spawned units are immediately selectable + commandable.
  - Insufficient minerals → queue is refused (no debit).

tests/test_production_pipeline.py — 6+ tests: queue debits, completion
spawns + places unit, insufficient minerals refused, multiple queued
items complete in order.
""",
            files=(
                "starcraft_hive/game.py",
                "starcraft_hive/entities/building.py",
                "tests/test_production_pipeline.py",
            ),
            criteria=(
                "queue debits minerals; refused if too poor",
                "completion spawns a controllable unit",
                "queued items finish in order",
            ),
            depends_on=("real_combat",),
            tags=("gameplay",),
            review_by=REVIEWER, polish_iters=3, smoke_cmd=sm,
        ),
        PlanTask(
            key="rally_points",
            title="Building rally points",
            body=_COMMON_RULES + """
Buildings can set a rally point that new units move to.
  - Building gains `rally: tuple[int,int] | None`. With a building
    selected, right-click sets its rally point (instead of a unit
    move). Newly produced units get state='moving' with target=rally.
  - render/world.py draws a faint rally flag/line from a selected
    building to its rally point.

tests/test_rally.py — 4+ tests: right-click with base selected sets
rally; produced unit heads to rally; clearing rally (right-click on
base) resets.
""",
            files=(
                "starcraft_hive/entities/building.py",
                "starcraft_hive/input/handler.py",
                "starcraft_hive/game.py",
                "starcraft_hive/render/world.py",
                "tests/test_rally.py",
            ),
            criteria=(
                "right-click sets a building rally point",
                "produced unit moves to rally",
            ),
            depends_on=("production_pipeline",),
            tags=("gameplay",),
            review_by=REVIEWER, polish_iters=2, smoke_cmd=sm,
        ),
        PlanTask(
            key="balance_pass",
            title="Balance pass: a match resolves in reasonable time",
            body=_COMMON_RULES + """
Tune data/races.py stats so a player-vs-AI match actually progresses
to a conclusion in a bounded number of frames (neither instant nor
never). Adjust worker mineral_carry + gather time, unit hp/damage/
range/speed, build times, and the enemy AI thresholds so that:
  - the economy reaches ~military-affordable within ~600 frames,
  - combats resolve (a fight ends with a winner within ~300 frames),
  - a full skirmish reaches game_over within ~5000 frames headless.

tests/test_balance.py — 4+ tests: worker gathers >=1 load within 600
frames; a staged 3v3 fight ends within 300 frames; affordability
timeline sane.
""",
            files=(
                "starcraft_hive/data/races.py",
                "tests/test_balance.py",
            ),
            criteria=(
                "economy affordable within ~600 frames",
                "staged fight resolves within ~300 frames",
            ),
            depends_on=("production_pipeline",),
            tags=("gameplay", "balance"),
            review_by=REVIEWER, polish_iters=3, smoke_cmd=sm,
        ),
        PlanTask(
            key="win_flow",
            title="Win/lose screen with restart + match stats",
            body=_COMMON_RULES + """
Polish the end-of-match flow.
  - render/hud.py draw_game_over shows VICTORY/DEFEAT plus match
    stats: units built, units lost, enemy units killed, minerals
    gathered, match duration (ticks/60).
  - GameState tracks those counters (increment on spawn/death/gather).
  - __main__: pressing R on the game-over screen restarts a fresh
    match (rebuild state + reassign harvesters); ESC quits.

tests/test_win_flow.py — 5+ tests: counters increment on spawn/kill/
gather; draw_game_over renders stats headless; restart produces a
fresh tick-0 state.
""",
            files=(
                "starcraft_hive/game.py",
                "starcraft_hive/render/hud.py",
                "starcraft_hive/__main__.py",
                "tests/test_win_flow.py",
            ),
            criteria=(
                "match stats counters tracked + rendered",
                "R restarts a fresh match",
            ),
            depends_on=("balance_pass",),
            tags=("gameplay", "ui"),
            review_by=REVIEWER, polish_iters=2, smoke_cmd=sm,
        ),
        PlanTask(
            key="gameplay_integration",
            title="Integration: full skirmish reaches a winner",
            body=_COMMON_RULES + """
End-to-end gameplay guard.
  - tests/test_full_skirmish.py: run a headless match (player AI +
    enemy AI both producing + attacking) for up to 5000 frames and
    assert game_over becomes 'player' or 'enemy' (a winner emerges) —
    OR, if 5000 frames is too tight after balancing, assert clear
    progress: both sides built military AND total unit deaths > 0 AND
    one side's building count dropped. Pick the strongest invariant
    the balanced game supports and document it.
  - tools/skirmish_report.py prints the final match stats.

Depends on every gameplay task so it runs last.
""",
            files=(
                "tests/test_full_skirmish.py",
                "tools/skirmish_report.py",
            ),
            criteria=(
                "headless skirmish reaches a winner or clear progress",
                "skirmish_report prints final stats",
            ),
            depends_on=(
                "real_combat", "production_pipeline", "rally_points",
                "balance_pass", "win_flow",
            ),
            tags=("gameplay", "integration"),
            review_by=REVIEWER, polish_iters=3, smoke_cmd=sm,
        ),
    ]


def _queue_gameplay() -> None:
    """Gameplay-depth tasks → board, play-loop smoke-gated."""
    _queue_tasks(_gameplay_plan, "gameplay")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--drive", action="store_true")
    ap.add_argument("--polish", action="store_true")
    ap.add_argument("--improve", action="store_true")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--gameplay", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if args.reset:
        import shutil
        if PROJECT_PATH.exists():
            shutil.rmtree(PROJECT_PATH, ignore_errors=True)
        store = CrewBoardStore(VAULT_DB)
        # Archive all sc-build tasks
        archived = 0
        for t in store.list_tasks():
            if "sc-build" in (t.tags or []):
                if t.status != schema.STATUS_ARCHIVED:
                    # Walk the state machine to archived
                    for intermediate in (
                        schema.STATUS_READY, schema.STATUS_IN_PROGRESS,
                        schema.STATUS_REVIEW, schema.STATUS_ARCHIVED,
                    ):
                        try:
                            store.move_task(t.slug, intermediate, actor="system",
                                            detail="reset")
                        except ValueError:
                            continue
                archived += 1
        print(f"reset: archived {archived} sc-build tasks; wiped project dir")
        return 0

    if args.bootstrap:
        _bootstrap()
        return 0
    if args.status:
        _print_status()
        return 0
    if args.polish:
        _queue_polish()
        return 0
    if args.improve:
        _queue_improvements()
        return 0
    if args.controls:
        _queue_controls()
        return 0
    if args.gameplay:
        _queue_gameplay()
        return 0
    if args.drive:
        return asyncio.run(_drive())
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
