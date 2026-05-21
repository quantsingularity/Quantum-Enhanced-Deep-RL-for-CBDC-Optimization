#!/usr/bin/env bash
# lint.sh
# -------
# Run from the project root or from scripts/.
# Lints the target directory with ruff, flake8, mypy, and pylint.

shopt -s globstar 2>/dev/null

# ── resolve project root ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── prompt ────────────────────────────────────────────────────────────────────
echo ""
echo "Available directories under code/:"
ls -d "$PROJECT_ROOT/code"/*/  2>/dev/null | xargs -I{} basename {}
echo ""
read -rp "Enter sub-directory to lint (relative to code/, or '.' for all of code/): " TARGET_REL

TARGET="$PROJECT_ROOT/code/${TARGET_REL}"

# ── validate ──────────────────────────────────────────────────────────────────
if [[ -z "$TARGET_REL" ]]; then
    echo "Error: no directory entered."
    exit 1
fi

if [[ ! -d "$TARGET" ]]; then
    echo "Error: '$TARGET' is not a valid directory."
    exit 1
fi

PY_FILES=$(find "$TARGET" -name "*.py" | head -1)
if [[ -z "$PY_FILES" ]]; then
    echo "No Python files found in '$TARGET'."
    exit 1
fi

PY_COUNT=$(find "$TARGET" -name "*.py" | wc -l | tr -d ' ')
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Target  : $TARGET"
echo "  Python files found: $PY_COUNT"
echo "══════════════════════════════════════════════════════════"

# ── ruff ──────────────────────────────────────────────────────────────────────
echo ""
echo "[ 1/4 ] ruff ─────────────────────────────────────────────"
if command -v ruff &>/dev/null; then
    ruff check "$TARGET" --fix
else
    echo "  ruff not installed — run: pip install ruff"
fi

# ── flake8 ────────────────────────────────────────────────────────────────────
echo ""
echo "[ 2/4 ] flake8 ───────────────────────────────────────────"
if command -v flake8 &>/dev/null; then
    flake8 "$TARGET" --max-line-length 100 --statistics
else
    echo "  flake8 not installed — run: pip install flake8"
fi

# ── mypy ──────────────────────────────────────────────────────────────────────
echo ""
echo "[ 3/4 ] mypy ─────────────────────────────────────────────"
if command -v mypy &>/dev/null; then
    mypy "$TARGET" --ignore-missing-imports
else
    echo "  mypy not installed — run: pip install mypy"
fi

# ── pylint ────────────────────────────────────────────────────────────────────
echo ""
echo "[ 4/4 ] pylint ───────────────────────────────────────────"
if command -v pylint &>/dev/null; then
    find "$TARGET" -name "*.py" | xargs pylint
else
    echo "  pylint not installed — run: pip install pylint"
fi

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Done."
echo "══════════════════════════════════════════════════════════"
