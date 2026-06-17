"""Document each image/video generation tool in the vault, with smart
[[wikilinks]] back to every compatible LoRA.

Companion to `seed_lora_notes.py`. Where the LoRA seeder writes a leaf
node per add-on, this script writes the *frame* — pipeline overviews,
checkpoint cards, and video-model cards — that ties those LoRAs together
so the librarian helper can answer "what tools do we have for X" without
hand-rolling a separate listing.

Sources of truth:
  - Image checkpoints: `C:\\Projects\\imageToVideo\\models\\community\\registry.json`
  - LoRAs (for cross-linking): `C:\\Projects\\imageToVideo\\models\\loras\\lora_registry.json`
  - Video tools: enumerated below (wired from on-disk model dirs +
    imageToVideo's video pipeline modules; small + stable enough to
    inline rather than parse a third registry)

Each note carries:
  - tags: [tool, <kind>, <pipeline?>] for filterable retrieval
  - extra.tool_kind: "image-pipeline" | "checkpoint" | "video-tool"
  - audience: [terry, claude-code]
  - body: what it is, when to use, **list of compatible LoRAs as
    [[wikilinks]]** (smart-linked off the alias the LoRA seeder used)

Idempotent. Same dedup behaviour as seed_lora_notes.py.

Usage:
  python scripts/seed_tool_notes.py
  python scripts/seed_tool_notes.py --dry-run
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

LORA_REGISTRY = Path(r"C:\Projects\imageToVideo\models\loras\lora_registry.json")
COMMUNITY_REGISTRY = Path(r"C:\Projects\imageToVideo\models\community\registry.json")
DEFAULT_URL = "http://127.0.0.1:8766"


# Pipeline overview blurbs. Hand-written because the gateway needs
# accurate "when to reach for it" guidance — the registry only carries
# the technical fields. Keep this list in sync with the pipelines that
# actually appear in `lora_registry.json` + `community/registry.json`.
PIPELINE_OVERVIEWS: dict[str, dict] = {
    "flux": {
        "title": "FLUX Pipeline (Image)",
        "summary": (
            "Black Forest Labs' FLUX family — high-fidelity transformer "
            "diffusion. Default for realistic portraits, complex prompts "
            "and long captions. Heavy on VRAM; the 4-bit and Lightning "
            "variants we keep installed exist to bring it onto a 24 GB card."
        ),
        "when": (
            "- realistic / photographic prompts\n"
            "- prompts longer than ~70 tokens (FLUX handles long prompts well)\n"
            "- when SDXL gives plastic skin or muddy lighting\n"
            "- Lightning 8-step variant when you need ~10 s renders"
        ),
        "checkpoints_blurb": "Installed checkpoints",
        "default_strength_note": "LoRAs typically apply at 0.6–1.0.",
    },
    "sdxl": {
        "title": "SDXL Pipeline (Image)",
        "summary": (
            "Stable Diffusion XL — the workhorse for stylised + NSFW "
            "imagery. Faster than FLUX on the same hardware, broader "
            "LoRA ecosystem (especially Pony-tuned models). Lower max "
            "fidelity but very strong at character/style control."
        ),
        "when": (
            "- stylised / illustrated prompts\n"
            "- NSFW prompts where Pony LoRAs make a difference\n"
            "- character-likeness prompts using SDXL likeness LoRAs\n"
            "- when render time matters more than raw fidelity"
        ),
        "checkpoints_blurb": "Installed checkpoints",
        "default_strength_note": "LoRAs typically apply at 0.7–1.0; sliders 0.3–0.6.",
    },
    "sd15": {
        "title": "SD1.5 Pipeline (Image)",
        "summary": (
            "Stable Diffusion 1.5 — the legacy backbone. Kept for the "
            "tail of likeness LoRAs that were never re-trained on SDXL "
            "or FLUX (older celebrities, niche subjects). Lower fidelity "
            "but unmatched LoRA breadth for specific people."
        ),
        "when": (
            "- the prompt names a person whose only LoRA is SD1.5\n"
            "- you want the classic SD1.5 'painted' look\n"
            "- never as a default — only when a specific SD1.5 LoRA is needed"
        ),
        "checkpoints_blurb": "Installed checkpoints",
        "default_strength_note": "LoRAs typically apply at 0.7–0.9.",
    },
    "zimage": {
        "title": "Z-Image Pipeline (Image)",
        "summary": (
            "Tongyi-MAI's Z-Image-Turbo — a fast, prompt-tuned diffusion "
            "model. Niche; sits alongside FLUX/SDXL but isn't the default "
            "for anything yet. Worth trying when FLUX is busy and you "
            "want quality between SDXL and FLUX."
        ),
        "when": (
            "- experimental fallback if FLUX queue is backed up\n"
            "- benchmarks vs. FLUX/SDXL on a given prompt"
        ),
        "checkpoints_blurb": "Installed checkpoints",
        "default_strength_note": "No mature LoRA ecosystem yet — apply with caution.",
    },
    "wan": {
        "title": "WAN Pipeline (Video)",
        "summary": (
            "Alibaba's Wan 2.2 — image-to-video diffusion. Drives short "
            "(~5 s) video clips from a still + prompt. The high-noise / "
            "low-noise transformer pair lives at "
            "`models/wan22_i2v/`; the enhanced-NSFW GGUF lives at "
            "`models/wan22_enhanced_nsfw/`."
        ),
        "when": (
            "- user wants a short animated clip from an image\n"
            "- 'make this move' / 'animate this' prompts\n"
            "- always image-to-video (no text-only video here)"
        ),
        "checkpoints_blurb": "Installed transformers",
        "default_strength_note": "Video LoRAs apply at 0.5–0.9 typically.",
    },
}


# Video tools that aren't "checkpoints" in the registry sense.
VIDEO_TOOLS: list[dict] = [
    {
        "title": "Video Tool — LTX 2.3 (text/image to video)",
        "kind": "video-tool",
        "pipeline": "ltx",
        "main_file": r"C:\Projects\imageToVideo\models\ltx-2.3-22b-distilled.safetensors",
        "summary": (
            "Lightricks LTX 2.3 22B distilled — the secondary video "
            "engine alongside WAN. Text-to-video and image-to-video "
            "both supported. Faster turnaround than WAN at the cost of "
            "some fidelity. Spatial + temporal upscalers ship alongside."
        ),
        "when": (
            "- user wants a quick clip with no input image\n"
            "- text-to-video prompts (WAN doesn't do those here)\n"
            "- when WAN's queue is busy"
        ),
        "files": [
            "ltx-2.3-22b-distilled.safetensors (main)",
            "ltx-2.3-22b-distilled-lora-384.safetensors (LoRA-tuned variant)",
            "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
            "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
        ],
    },
    {
        "title": "Video Tool — WAN 2.2 i2v (image-to-video)",
        "kind": "video-tool",
        "pipeline": "wan",
        "main_file": r"C:\Projects\imageToVideo\models\wan22_i2v",
        "summary": (
            "Wan 2.2 image-to-video diffusion pipeline. Two-stage "
            "transformer (high-noise + low-noise) with a shared "
            "scheduler. Loads from `models/wan22_i2v/` via the "
            "diffusers pipeline. ~5 s clips at 480p/720p."
        ),
        "when": (
            "- user has an image they want animated\n"
            "- 'make her wave' / 'pan across this scene'\n"
            "- subject motion, not full text-to-video"
        ),
        "files": [
            "transformer/ + transformer_2/ (two-stage diffusion)",
            "vae/, scheduler/, text_encoder/, tokenizer/",
            "transformer_bf16.safetensors + transformer_2_bf16.safetensors (single-file fallback)",
        ],
    },
    {
        "title": "Video Tool — WAN 2.2 Enhanced NSFW (GGUF)",
        "kind": "video-tool",
        "pipeline": "wan",
        "main_file": r"C:\Projects\imageToVideo\models\wan22_enhanced_nsfw\wan22EnhancedNSFWSVICamera_nolightningSVICfQ8H.gguf",
        "summary": (
            "Quantised GGUF variant of WAN 2.2 fine-tuned on adult / "
            "intimate content. Smaller VRAM footprint than the BF16 "
            "transformers but lower precision. Use when adult-flavoured "
            "i2v is requested and VRAM is constrained."
        ),
        "when": (
            "- adult-content image-to-video requests\n"
            "- when full-precision wan22_i2v won't fit alongside other models\n"
            "- pairs naturally with the WAN-pipeline LoRAs (creampie, side-view, etc.)"
        ),
        "files": [
            "wan22EnhancedNSFWSVICamera_nolightningSVICfQ8H.gguf (single-file Q8)",
        ],
    },
]


def _slug(s: str) -> str:
    """Match the gateway's title→slug behaviour for wikilink targets."""
    out = []
    last_dash = False
    for ch in s.lower().strip():
        if ch.isalnum():
            out.append(ch)
            last_dash = False
        elif not last_dash:
            out.append("-")
            last_dash = True
    return "".join(out).strip("-")


def _pair(base_url: str, name: str = "tool-seed") -> tuple[str, str]:
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


def _load_registries() -> tuple[list[dict], list[dict]]:
    if not LORA_REGISTRY.exists():
        print(f"error: lora registry missing at {LORA_REGISTRY}")
        sys.exit(2)
    if not COMMUNITY_REGISTRY.exists():
        print(f"error: community registry missing at {COMMUNITY_REGISTRY}")
        sys.exit(2)
    loras = json.loads(LORA_REGISTRY.read_text(encoding="utf-8"))
    checkpoints = json.loads(COMMUNITY_REGISTRY.read_text(encoding="utf-8"))
    return loras, checkpoints


def _loras_for_pipeline(loras: list[dict], pipeline: str) -> list[dict]:
    return [l for l in loras if (l.get("pipeline") or "").lower() == pipeline]


def _format_lora_links(loras: list[dict]) -> str:
    if not loras:
        return "_(none installed for this pipeline)_"
    # Group by category for readability — sliders, characters, styles, ...
    groups: dict[str, list[dict]] = {}
    for l in loras:
        cat = (l.get("category") or "general").lower()
        groups.setdefault(cat, []).append(l)
    lines = []
    for cat in sorted(groups):
        bucket = sorted(groups[cat], key=lambda x: x.get("alias", "").lower())
        items = ", ".join(f"[[{l['alias']}]]" for l in bucket if l.get("alias"))
        lines.append(f"- **{cat}** — {items}")
    return "\n".join(lines)


def _build_pipeline_body(pipe: str, info: dict, loras: list[dict],
                        checkpoints: list[dict]) -> str:
    matched_ckpts = [c for c in checkpoints
                     if (c.get("pipeline") or "").lower() == pipe]
    ckpt_lines = "\n".join(
        f"- **{c.get('alias') or c.get('repo_id')}** — `{c.get('repo_id')}`"
        for c in matched_ckpts
    ) or "_(no checkpoints registered)_"
    parts = [
        f"**Pipeline ID:** `{pipe}`",
        f"**LoRAs installed:** {len(loras)}",
        f"**Checkpoints installed:** {len(matched_ckpts)}",
        "",
        "## What it is",
        info["summary"],
        "",
        "## When to reach for it",
        info["when"],
        "",
        f"## {info['checkpoints_blurb']}",
        ckpt_lines,
        "",
        "## Compatible LoRAs",
        info["default_strength_note"],
        "",
        _format_lora_links(loras),
    ]
    return "\n".join(parts) + "\n"


def _build_checkpoint_body(entry: dict, loras: list[dict]) -> str:
    pipe = (entry.get("pipeline") or "unknown").lower()
    matched = _loras_for_pipeline(loras, pipe) if pipe in PIPELINE_OVERVIEWS else []
    pipeline_link = (
        f"[[{PIPELINE_OVERVIEWS[pipe]['title']}]]"
        if pipe in PIPELINE_OVERVIEWS else f"`{pipe}`"
    )
    parts = [
        f"**Type:** checkpoint",
        f"**Pipeline:** {pipeline_link}",
        f"**Repo ID:** `{entry.get('repo_id')}`",
        f"**Local path:** `{entry.get('local_path')}`",
        "",
        "## How to use",
        f"Pick this checkpoint when the user wants the **{entry.get('alias')}** "
        f"look. It runs through the {pipe.upper()} pipeline, so any "
        f"`{pipe}` LoRA can be stacked on top.",
        "",
        "## Compatible LoRAs (top by category)",
        _format_lora_links(matched[:30]) if matched else
        "_(no LoRAs registered for this pipeline yet)_",
    ]
    return "\n".join(parts) + "\n"


def _build_video_body(tool: dict, loras: list[dict]) -> str:
    pipe = tool["pipeline"]
    matched = _loras_for_pipeline(loras, pipe) if pipe in PIPELINE_OVERVIEWS else []
    file_lines = "\n".join(f"- `{f}`" for f in tool.get("files") or [])
    pipeline_link = (
        f"[[{PIPELINE_OVERVIEWS[pipe]['title']}]]"
        if pipe in PIPELINE_OVERVIEWS else f"`{pipe}`"
    )
    parts = [
        f"**Type:** video tool",
        f"**Pipeline:** {pipeline_link}",
        f"**Main path:** `{tool['main_file']}`",
        "",
        "## What it is",
        tool["summary"],
        "",
        "## When to reach for it",
        tool["when"],
        "",
        "## Files on disk",
        file_lines or "_(single file)_",
    ]
    if matched:
        parts.extend([
            "",
            "## Compatible LoRAs",
            _format_lora_links(matched),
        ])
    return "\n".join(parts) + "\n"


def _post(base_url: str, token: str, *, title: str, body: str,
          tags: list[str], extra: dict, dry_run: bool) -> tuple[bool, str]:
    payload = {
        "category": "knowledge",
        "title": title,
        "body": body,
        "audience": ["terry", "claude-code"],
        "tags": tags,
        "extra": extra,
    }
    if dry_run:
        return True, f"[dry] would POST: {title} ({len(body)} chars, tags={tags})"
    try:
        resp = _http_post_json(f"{base_url}/v1/vault/learn", payload, token=token)
    except urllib.error.HTTPError as e:
        body_err = e.read().decode("utf-8", errors="replace")[:160]
        return False, f"http {e.code}: {body_err}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    path = resp.get("path") if isinstance(resp, dict) else None
    return True, f"saved {path}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    loras, checkpoints = _load_registries()

    # Build the queue of (title, body, tags, extra) tuples up front so
    # we can show a count before pacing.
    queue: list[tuple[str, str, list[str], dict]] = []

    # 1) Pipeline overview notes
    for pipe, info in PIPELINE_OVERVIEWS.items():
        pipe_loras = _loras_for_pipeline(loras, pipe)
        body = _build_pipeline_body(pipe, info, pipe_loras, checkpoints)
        queue.append((
            info["title"],
            body,
            ["tool", "image-pipeline" if pipe != "wan" else "video-pipeline", pipe],
            {"tool_kind": "image-pipeline" if pipe != "wan" else "video-pipeline",
             "pipeline": pipe, "lora_count": len(pipe_loras)},
        ))

    # 2) Checkpoint notes
    for ckpt in checkpoints:
        alias = ckpt.get("alias") or ckpt.get("repo_id")
        if not alias:
            continue
        pipe = (ckpt.get("pipeline") or "unknown").lower()
        title = f"Checkpoint — {alias}"
        body = _build_checkpoint_body(ckpt, loras)
        tags = ["tool", "checkpoint"]
        if pipe in PIPELINE_OVERVIEWS:
            tags.append(pipe)
        queue.append((
            title, body, tags,
            {"tool_kind": "checkpoint", "pipeline": pipe,
             "repo_id": ckpt.get("repo_id"), "local_path": ckpt.get("local_path")},
        ))

    # 3) Video tool notes
    for vtool in VIDEO_TOOLS:
        body = _build_video_body(vtool, loras)
        queue.append((
            vtool["title"], body,
            ["tool", "video", vtool["pipeline"]],
            {"tool_kind": "video-tool", "pipeline": vtool["pipeline"],
             "main_file": vtool["main_file"]},
        ))

    print(f"# {len(queue)} tool notes to write "
          f"({len(PIPELINE_OVERVIEWS)} pipelines + "
          f"{len(checkpoints)} checkpoints + "
          f"{len(VIDEO_TOOLS)} video tools)")

    token = device_id = ""
    if not args.dry_run:
        token, device_id = _pair(args.url)
        print(f"# paired (device {device_id[:10]})")

    ok = err = 0
    rate_window: list[float] = []
    try:
        for i, (title, body, tags, extra) in enumerate(queue, start=1):
            success, note = _post(
                args.url, token,
                title=title, body=body, tags=tags, extra=extra,
                dry_run=args.dry_run,
            )
            if success:
                ok += 1
            else:
                err += 1
            print(f"  {i:>3}/{len(queue)}  {title[:48]:<48}  {note}")
            if not args.dry_run and i < len(queue):
                rate_window.append(time.monotonic())
                if len(rate_window) >= 30:
                    elapsed = time.monotonic() - rate_window[0]
                    if elapsed < 60:
                        sleep_for = 60 - elapsed + 0.5
                        print(f"  pacing: sleep {sleep_for:.1f}s "
                              f"(30 writes in {elapsed:.1f}s)")
                        time.sleep(sleep_for)
                    rate_window = []
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
