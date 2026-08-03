# Portable clangd setup (cross-server)

Go-to-definition for C++ in Cursor Remote-SSH is **per host**. Use this script once on each new server.

## One command (any SSH host)

```bash
# from the shared workspace volume
bash /Volumes/case_sensitive_workspace/setup_clangd_cursor.sh

# or the copy tracked in this branch
bash workspace/setup_clangd_cursor.sh

# or pass an explicit triton checkout
bash workspace/setup_clangd_cursor.sh /path/to/triton
```

Then in Cursor: **Developer: Reload Window**.

## What it does on each host

1. Installs `clangd` if missing (`apt`)
2. Installs extension `llvm-vs-code-extensions.vscode-clangd` via Cursor remote CLI
3. Writes remote user settings enabling clangd (disables Microsoft C/C++ IntelliSense conflict)
4. Writes portable `triton/.clangd` that uses repo-root `compile_commands.json`
5. Links `compile_commands.json` from the cmake build dir if needed

## Jump keys

- **F12** / **Ctrl+click** — Go to Definition
- **Alt+F12** — Peek Definition
- **Shift+F12** — References

## Requirements

- Folder opened via **Cursor Remote-SSH** (so remote CLI + extensions exist)
- A CMake build with `compile_commands.json` (Triton usually has this under `build/cmake.*`)

## Verify

```bash
clangd --version
clangd --check=lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp
```

Expect `All checks completed` (warnings about missing system headers are OK if definitions still resolve).
