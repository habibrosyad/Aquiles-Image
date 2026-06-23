#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") <site-packages-path>

Overwrite the installed aquiles-image pip package with the current source build.

Arguments:
  <site-packages-path>   Path to the Python site-packages directory where
                         aquiles-image is installed (e.g., .../lib/python3.11/site-packages)

Alternatively, pass a Python interpreter path and the script will resolve
its site-packages automatically:

  $(basename "$0") /path/to/venv/bin/python3
  $(basename "$0") /usr/bin/python3

Examples:
  $(basename "$0") /Users/user/myenv/lib/python3.11/site-packages
  $(basename "$0") ~/.pyenv/versions/3.11.5/envs/myenv/bin/python
EOF
  exit 1
}

TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
  usage
fi

# --- Resolve the site-packages path ---
resolve_site_packages() {
  local input="$1"

  # If it looks like a Python executable, ask it for site-packages
  if [[ -f "$input" && -x "$input" ]]; then
    local sp
    sp=$("$input" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
    if [[ -z "$sp" ]]; then
      # Fallback: try from distutils
      sp=$("$input" -c "import sysconfig; print(sysconfig.get_path('purelib'))" 2>/dev/null || true)
    fi
    if [[ -n "$sp" && -d "$sp" ]]; then
      echo "$sp"
      return 0
    fi
  fi

  # If it contains "site-packages" in the path, use directly
  if [[ "$input" == *"site-packages"* ]]; then
    if [[ -d "$input" ]]; then
      echo "$input"
      return 0
    fi
  fi

  return 1
}

SP_DIR=""
if SP_DIR=$(resolve_site_packages "$TARGET"); then
  true  # already resolved
else
  # Maybe it's a site-packages dir directly?
  if [[ -d "$TARGET" ]]; then
    SP_DIR="$TARGET"
  else
    echo "ERROR: Cannot resolve site-packages from: $TARGET"
    echo ""
    usage
  fi
fi

echo "=== Target site-packages: $SP_DIR ==="

# --- Build the wheel ---
BUILD_DIR=$(mktemp -d)
cleanup() { rm -rf "$BUILD_DIR"; }
trap cleanup EXIT

echo "=== Building wheel from source ==="
cd "$SCRIPT_DIR"
if python -m build --wheel --outdir "$BUILD_DIR" >/dev/null 2>&1; then
  true
elif pip wheel --no-deps . -w "$BUILD_DIR" >/dev/null 2>&1; then
  true
else
  echo "ERROR: build failed. Install with: pip install build"
  exit 1
fi

WHEEL=$(ls "$BUILD_DIR"/*.whl 2>/dev/null | head -1)
if [[ -z "$WHEEL" ]]; then
  echo "ERROR: No wheel produced by build."
  exit 1
fi
echo "=== Built: $(basename "$WHEEL") ==="

# --- Extract wheel directly into site-packages, overwriting ---
echo "=== Installing into $SP_DIR ==="
# Remove old package directory and egg-info/dist-info
rm -rf "$SP_DIR/aquilesimage"
rm -f "$SP_DIR/aquiles_image"*.egg-link 2>/dev/null || true
rm -rf "$SP_DIR/aquiles_image"*.dist-info 2>/dev/null || true
rm -rf "$SP_DIR/aquiles_image"*.egg-info 2>/dev/null || true

# Extract the wheel
unzip -qo "$WHEEL" -d "$SP_DIR"

echo "=== Verifying ==="
if [[ -d "$SP_DIR/aquilesimage" ]]; then
  echo "OK: aquilesimage package files installed at $SP_DIR/aquilesimage"
else
  echo "WARNING: aquilesimage directory not found after extraction."
fi

# Try to import it to verify
PYTHON_BIN=$(cd "$SP_DIR" && find . -maxdepth 4 -name "python3*" -type f 2>/dev/null | head -1 || true)
if [[ -z "$PYTHON_BIN" ]]; then
  # Try to find it relative to site-packages
  PYTHON_BIN=$(dirname "$(dirname "$SP_DIR")")/bin/python3
fi
if [[ -f "$PYTHON_BIN" ]]; then
  echo "=== Testing import ==="
  "$PYTHON_BIN" -c "import aquilesimage; print('aquilesimage import OK'); print('version:', getattr(aquilesimage, '__version__', 'unknown'))" 2>&1 || echo "Import check skipped (version not exported)"
fi

echo ""
echo "Done. Package overwritten successfully."
