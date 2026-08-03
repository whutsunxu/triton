#!/usr/bin/env bash
# Portable Cursor C++ go-to-definition setup (clangd) for any SSH host.
# Usage:
#   ./setup_clangd_cursor.sh [TRITON_REPO_DIR]
# Default repo: /Volumes/case_sensitive_workspace/triton
set -euo pipefail

REPO="${1:-/Volumes/case_sensitive_workspace/triton}"
EXT_ID="llvm-vs-code-extensions.vscode-clangd"

echo "==> Repo: $REPO"

find_cursor_cli() {
  if command -v cursor >/dev/null 2>&1; then
    command -v cursor
    return
  fi
  if command -v code >/dev/null 2>&1; then
    command -v code
    return
  fi
  local cand
  cand=$(ls -d "$HOME"/.cursor-server/bin/linux-x64/*/bin/remote-cli/cursor 2>/dev/null | sort | tail -1 || true)
  if [[ -n "${cand}" && -x "${cand}" ]]; then
    echo "${cand}"
    return
  fi
  return 1
}

echo "==> Ensuring clangd binary"
if ! command -v clangd >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq clangd
  else
    echo "ERROR: clangd not found and apt-get unavailable. Install clangd manually." >&2
    exit 1
  fi
fi
clangd --version | head -1

echo "==> Installing Cursor extension: $EXT_ID"
CLI=$(find_cursor_cli) || {
  echo "ERROR: Cursor/code CLI not found. Open this folder in Cursor Remote-SSH first, then re-run." >&2
  exit 1
}
echo "    using CLI: $CLI"
"$CLI" --install-extension "$EXT_ID" || true

echo "==> Writing Cursor user settings (remote)"
USER_SETTINGS="$HOME/.cursor-server/data/User/settings.json"
mkdir -p "$(dirname "$USER_SETTINGS")"
python3 - <<'PY' "$USER_SETTINGS"
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
data = {}
if path.exists():
    try:
        data = json.loads(path.read_text())
    except Exception:
        data = {}
data.update({
    "clangd.path": "clangd",
    "clangd.arguments": [
        "--background-index",
        "--clang-tidy=false",
        "--header-insertion=never",
        "--completion-style=detailed",
    ],
    "clangd.onConfigChanged": "restart",
    "[cpp]": {
        "editor.defaultFormatter": "llvm-vs-code-extensions.vscode-clangd",
        "editor.suggest.insertMode": "replace",
    },
    "[c]": {
        "editor.defaultFormatter": "llvm-vs-code-extensions.vscode-clangd",
        "editor.suggest.insertMode": "replace",
    },
    # Avoid fighting clangd if Microsoft C/C++ is ever installed
    "C_Cpp.intelliSenseEngine": "disabled",
    "C_Cpp.autocomplete": "disabled",
    "C_Cpp.errorSquiggles": "disabled",
})
path.write_text(json.dumps(data, indent=2) + "\n")
print(f"    wrote {path}")
PY

echo "==> Writing portable repo clangd config"
mkdir -p "$REPO/.vscode"

cat > "$REPO/.clangd" <<'EOF'
# Portable: use compile_commands.json next to this file (repo root).
# Create/update it after cmake configure, e.g.:
#   ln -sfn build/cmake.*/compile_commands.json compile_commands.json
CompileFlags:
  CompilationDatabase: .
EOF

cat > "$REPO/.vscode/settings.json" <<'EOF'
{
  "clangd.path": "clangd",
  "clangd.arguments": [
    "--background-index",
    "--header-insertion=never"
  ],
  "[cpp]": {
    "editor.defaultFormatter": "llvm-vs-code-extensions.vscode-clangd"
  },
  "[c]": {
    "editor.defaultFormatter": "llvm-vs-code-extensions.vscode-clangd"
  },
  "C_Cpp.intelliSenseEngine": "disabled"
}
EOF

echo "==> Ensuring compile_commands.json at repo root"
if [[ ! -e "$REPO/compile_commands.json" ]]; then
  CCDB=$(find "$REPO/build" -name compile_commands.json 2>/dev/null | head -1 || true)
  if [[ -n "$CCDB" ]]; then
    ln -sfn "$CCDB" "$REPO/compile_commands.json"
    echo "    linked $REPO/compile_commands.json -> $CCDB"
  else
    echo "    WARN: no compile_commands.json found under $REPO/build"
    echo "    After cmake, run: ln -sfn <build>/compile_commands.json $REPO/compile_commands.json"
  fi
else
  echo "    found $REPO/compile_commands.json"
fi

echo
echo "Done. On THIS host:"
echo "  1) Command Palette -> Developer: Reload Window"
echo "  2) Open a .cpp file, use F12 / Ctrl+click for Go to Definition"
echo
echo "On a NEW SSH server, re-run this same script once."
