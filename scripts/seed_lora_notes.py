"""Write one vault note per installed LoRA so Hive can discover them
via librarian queries. Reads `lora_registry.json` and POSTs to the
gateway's `/v1/vault/learn` endpoint (so the autolinker, dedup, and
quality gate all run). Idempotent — re-running over notes already
written via this script bumps the `updated` field but doesn't
duplicate, because the dedup checker in action_executor merges
identical-title writes.

Each note lives at `knowledge/loras/<slug>.md` with frontmatter:
  - title:        the LoRA alias
  - audience:     [terry, claude-code]
  - tags:         [lora, <pipeline>, <category>, "nsfw" if flagged]
  - extra:        repo_id, default_strength, main_file, registry_path

Body sections:
  - **Pipeline**, **Category**, **NSFW**: facts.
  - **Trigger words**: the verbatim trigger string.
  - **How to use**: a one-line description of when to reach for it.
  - **File**: the `.safetensors` path on disk.
  - **Source**: the Civitai / HuggingFace URL when available.

Usage:
  python scripts/seed_lora_notes.py             # all 77 LoRAs, default URL
  python scripts/seed_lora_notes.py --dry-run   # print, don't post
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REGISTRY_PATH = Path(r"C:\Projects\imageToVideo\models\loras\lora_registry.json")
DEFAULT_URL = "http://127.0.0.1:8766"


def _pair(base_url: str, name: str = "lora-seed") -> str:
    """Return a fresh bearer token. Uses urllib so this script doesn't
    drag a third-party dep just to run a one-shot seed."""
    code_resp = _http_get_json(f"{base_url}/v1/pair/new")
    code = code_resp["code"]
    pair_resp = _http_post_json(
        f"{base_url}/v1/pair",
        {"code": code, "name": name, "platform": "smoke"},
    )
    return pair_resp["token"], pair_resp["device_id"]


def _http_get_json(url: str) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def _http_post_json(url: str, body: dict, *, token: str | None = None) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def _http_delete(url: str, *, token: str) -> int:
    req = urllib.request.Request(url, method="DELETE")
    req.add_header("Authorization", f"Bearer {token}")
    with contextlib.suppress(urllib.error.HTTPError):
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    return 0


def _usage_hint(entry: dict) -> str:
    """Human-readable 'when to reach for it' line, derived from the
    registry's category + NSFW flag + trigger style. Stays generic —
    each LoRA's curator wrote the trigger; we just frame the use."""
    pipeline = (entry.get("pipeline") or "unknown").lower()
    cat = (entry.get("category") or "general").lower()
    nsfw = bool(entry.get("nsfw"))
    alias = entry.get("alias", "")
    triggers = entry.get("trigger_words", "")
    base = f"Pulls in the {alias} ({pipeline}) LoRA"
    if cat in {"slider", "enhancement"}:
        base = (
            f"`{alias}` is a {pipeline} {cat} — apply it to nudge a "
            f"single visual axis (proportions / lighting / detail) "
            f"without overwhelming the rest of the prompt"
        )
    elif cat in {"character", "person", "celebrity"}:
        base = (
            f"`{alias}` is a {pipeline} likeness LoRA — use it when "
            f"the user has asked for that specific subject. Drop the "
            f"trigger token verbatim early in the prompt"
        )
    elif cat in {"style", "look", "aesthetic"}:
        base = (
            f"`{alias}` is a {pipeline} style LoRA — apply when the "
            f"user wants a specific visual mood / film-look / period "
            f"feel rather than a subject change"
        )
    elif cat == "nsfw" or nsfw:
        base = (
            f"`{alias}` is a {pipeline} NSFW LoRA — only apply when "
            f"the user has explicitly asked for adult content"
        )
    if triggers:
        base += f". Trigger: `{triggers}`"
    return base + "."


def _build_body(entry: dict) -> str:
    pipeline = entry.get("pipeline") or "unknown"
    category = entry.get("category") or "general"
    triggers = entry.get("trigger_words") or ""
    strength = entry.get("default_strength") or "1.0"
    main_file = entry.get("main_file") or "(unknown)"
    repo_id = entry.get("repo_id") or "(unknown)"
    nsfw = "yes" if entry.get("nsfw") else "no"

    source = ""
    if isinstance(repo_id, str) and repo_id.startswith("civitai:"):
        mid = repo_id.split(":", 1)[1].split("/", 1)[0]
        source = f"https://civitai.com/models/{mid}"
    elif isinstance(repo_id, str) and "/" in repo_id:
        source = f"https://huggingface.co/{repo_id}"

    parts = [
        f"**Pipeline:** {pipeline}",
        f"**Category:** {category}",
        f"**NSFW:** {nsfw}",
        f"**Default strength:** {strength}",
        "",
        "## Trigger words",
        f"`{triggers}`" if triggers else "_(none — apply by alias only)_",
        "",
        "## How to use",
        _usage_hint(entry),
        "",
        "## File",
        f"`{main_file}`",
    ]
    if source:
        parts.extend(["", "## Source", source])
    return "\n".join(parts) + "\n"


def _post_note(base_url: str, token: str, entry: dict, *, dry_run: bool) -> tuple[bool, str]:
    alias = (entry.get("alias") or "").strip()
    if not alias:
        return False, "no alias"
    pipeline = (entry.get("pipeline") or "unknown").lower()
    category = (entry.get("category") or "general").lower()
    tags = ["lora", pipeline]
    if category and category != pipeline:
        tags.append(category)
    if entry.get("nsfw"):
        tags.append("nsfw")
    body = _build_body(entry)
    payload = {
        "category": "knowledge",
        "title": alias,
        "body": body,
        "audience": ["terry", "claude-code"],
        "tags": tags,
        "extra": {
            "repo_id": entry.get("repo_id"),
            "main_file": entry.get("main_file"),
            "default_strength": entry.get("default_strength"),
            "pipeline": pipeline,
            "lora_doc": True,
        },
    }
    if dry_run:
        return True, f"[dry] would POST: {alias} ({len(body)} chars, tags={tags})"
    try:
        resp = _http_post_json(
            f"{base_url}/v1/vault/learn", payload, token=token,
        )
    except urllib.error.HTTPError as e:
        body_err = e.read().decode("utf-8", errors="replace")[:120]
        return False, f"http {e.code}: {body_err}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    path = resp.get("path") if isinstance(resp, dict) else None
    return True, f"saved {path}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0,
                   help="cap LoRAs for a quick test (0 = all)")
    args = p.parse_args(argv)

    if not REGISTRY_PATH.exists():
        print(f"error: registry missing at {REGISTRY_PATH}")
        return 2
    entries = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        print("error: registry isn't a list")
        return 2
    if args.limit:
        entries = entries[: args.limit]

    print(f"# {len(entries)} LoRAs to document")

    token = device_id = ""
    if not args.dry_run:
        token, device_id = _pair(args.url)
        print(f"# paired (device {device_id[:10]})")

    ok = err = 0
    rate_limit_window: list[float] = []
    try:
        for i, entry in enumerate(entries, start=1):
            # Pace under writes_per_minute (~60/min default). Sleep
            # 1.2s between posts to stay comfortably under.
            success, note = _post_note(
                args.url, token, entry, dry_run=args.dry_run,
            )
            if success:
                ok += 1
            else:
                err += 1
            print(f"  {i:>3}/{len(entries)}  "
                  f"{(entry.get('alias') or '')[:30]:<30}  {note}")
            if not args.dry_run and i < len(entries):
                # Rate-limit floor: max 30 writes/min so we never hit
                # the 60/min ceiling and trip 429.
                rate_limit_window.append(time.monotonic())
                if len(rate_limit_window) >= 30:
                    elapsed = time.monotonic() - rate_limit_window[0]
                    if elapsed < 60:
                        sleep_for = 60 - elapsed + 0.5
                        print(f"  pacing: sleep {sleep_for:.1f}s "
                              f"(30 writes in {elapsed:.1f}s)")
                        time.sleep(sleep_for)
                    rate_limit_window = []
                else:
                    time.sleep(1.2)
    finally:
        if not args.dry_run and token and device_id:
            _http_delete(
                f"{args.url}/v1/devices/{device_id}", token=token,
            )

    print(f"# done: {ok} ok, {err} err")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
