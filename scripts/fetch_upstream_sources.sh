#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/third_party"
python3 - "$ROOT" "$@" <<'PY'
import json, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1]); requested = set(sys.argv[2:])
sources = json.loads((root / 'third_party' / 'sources.json').read_text())
names = {s['name'] for s in sources}
if requested and not requested <= names:
    sys.exit(f"Unknown repositories: {sorted(requested - names)}. Choices: {sorted(names)}")
for src in sources:
    if requested and src['name'] not in requested:
        continue
    target = root / 'third_party' / src['name']
    if target.exists():
        print(f'SKIP {target} (exists)')
        continue
    print(f"CLONE {src['url']} -> {target}", flush=True)
    subprocess.run(['git','clone','--depth','1','--filter=blob:none',src['url'],str(target)], check=True)
    sha = subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'], text=True).strip()
    print(f'{src["name"]}: {sha}', flush=True)
PY
