#!/usr/bin/env bash
# mirror-sync.sh — deterministic private→public transform for the artificer mirror.
#
# Copies framework files from a private subtree into the public tree, applying:
#   1. CURATION — skip EXCLUDE paths (personal-only features that never ship public).
#   2. DE-ALIAS — private model aliases → public catalog aliases.
#   3. SCRUB    — owner PII (name/email/ids/IPs/paths/personal-projects) → generic/env.
#
# Copies .py/.md/.html/.js/.ts/.tsx/.css only; never .env/state/*.db/__pycache__.
# Does NOT delete public-only files. Idempotent. DRY-RUN unless --apply.
# After --apply, ALWAYS run scripts/release/check-all.sh + gitleaks before commit.
#
# Usage: mirror-sync.sh --src <private_repo> --rel <subpath> [--dest-rel <subpath>] [--apply]
#   gateway:   mirror-sync.sh --src /c/Projects/Ai-Team --rel gateway --apply
#   dashboard: mirror-sync.sh --src /c/Projects/hive-dashboard --rel src --dest-rel dashboard/src --apply
set -uo pipefail

PUBLIC_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

SRC="" ; REL="" ; DEST_REL="" ; APPLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --src) SRC="$2"; shift 2;;
    --rel) REL="$2"; shift 2;;
    --dest-rel) DEST_REL="$2"; shift 2;;
    --apply) APPLY=1; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done
[ -n "$SRC" ] && [ -n "$REL" ] || { echo "need --src and --rel"; exit 2; }
[ -n "$DEST_REL" ] || DEST_REL="$REL"

SRC_DIR="$SRC/$REL"
DEST_DIR="$PUBLIC_ROOT/$DEST_REL"
[ -d "$SRC_DIR" ] || { echo "no src dir: $SRC_DIR"; exit 2; }

# ── CURATION: paths (relative to --rel) the sync must NOT overwrite ──
#  - personal-only features that never ship public, AND
#  - files PUBLIC deliberately env-ified (HIVE_* vars) — preserve that work
#    rather than re-clobber with hardcoded private values.
#  - files PUBLIC already SCRUBBED for NSFW/uncensored-model content (preserve).
EXCLUDE_RE='^(routes/appstore\.py|routes/app_update\.py|tests/test_appstore\.py|action_executor\.py|config\.py|hive_turn_helpers\.py|image_shim\.py|video_shim\.py|routes/proactive\.py|routes/suno\.py|sandbox/python_runtime\.py|tests/test_proactive_hive\.py|tests/test_terminal\.py|asset_importer\.py|safe_fetcher\.py|orchestrator/tests/test_bench_harness\.py|tests/test_model_catalog\.py|tests/test_asset_importer\.py|tests/test_recipes_route\.py|tests/test_recipe_store\.py|tests/test_safe_fetcher\.py)$'

# ── transform: applied to every copied text file (sed -i). Order matters:
#    most-specific PII first so the owner-name pass also cleans owner paths. ──
apply_transform() {
  local f="$1"
  sed -i -E \
    -e 's/hive-qwen/planner-qwen/g' \
    -e 's/qwen35-claude/coder-qwen/g' \
    -e 's/maggy-qwen/helper-qwen/g' \
    -e 's/qwen3\.6:27b/qwen2.5-coder:7b/g' \
    -e 's/100\.79\.196\.101/127.0.0.1/g' \
    -e 's/10\.0\.0\.(33|172)/127.0.0.1/g' \
    -e 's/HIVE-RIG/localhost/g' \
    -e 's/345965297704108033|1490394153455194293/000000000000000000/g' \
    -e 's/RFCY61PEJPR/DEVICE_ID/Ig' \
    -e 's/davids-z-fold7/android-device/Ig' \
    -e 's#/Users/David Smereski/\.ssh#/tmp/.ssh#Ig' \
    -e 's#/Users/David Smereski#/tmp#Ig' \
    -e 's/David Smereski/Sample User/Ig' \
    -e 's/dsmereski/sampleuser/Ig' \
    -e 's/smereski-casino/example-app/Ig' \
    -e 's/\bsmereski\b/Sample/Ig' \
    -e 's/E:\\ollama/D:\\models/Ig' \
    -e 's/freedomguard/example-app/Ig' \
    -e 's/chesmeski/example-app/Ig' \
    -e 's/or9\.space/example.com/Ig' \
    "$f"
}

copied=0 ; skipped=0 ; transformed=0
while IFS= read -r src_file; do
  rel="${src_file#"$SRC_DIR"/}"
  if echo "$rel" | grep -qE "$EXCLUDE_RE"; then
    echo "  SKIP (personal): $REL/$rel" ; skipped=$((skipped+1)) ; continue
  fi
  copied=$((copied+1))
  if [ "$APPLY" -eq 1 ]; then
    mkdir -p "$DEST_DIR/$(dirname "$rel")"
    cp "$src_file" "$DEST_DIR/$rel"
    apply_transform "$DEST_DIR/$rel" && transformed=$((transformed+1))
  fi
done < <(find "$SRC_DIR" -type f \
  \( -name '*.py' -o -name '*.md' -o -name '*.html' -o -name '*.js' -o -name '*.ts' -o -name '*.tsx' -o -name '*.css' \) \
  -not -path '*__pycache__*' -not -path '*.pytest_cache*' -not -path '*node_modules*' -not -path '*/dist/*')

echo "──────────────────────────────────────"
echo "src=$REL  dest=$DEST_REL  apply=$APPLY"
echo "copied=$copied  skipped_personal=$skipped  transformed=$transformed"
[ "$APPLY" -eq 0 ] && echo "(dry-run — pass --apply to write)"
echo "NEXT: scripts/release/check-all.sh .  &&  gitleaks detect  (before commit)"
