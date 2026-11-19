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
    expected = src['sha']
    if target.exists():
        sha = subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'], text=True).strip()
        if sha != expected:
            sys.exit(f"Source {src['name']} is at {sha}, expected {expected}; inspect before replacing")
        print(f"VERIFIED {src['name']} at {sha}", flush=True)
        continue
    print(f"CLONE pinned {src['url']} at {expected}", flush=True)
    subprocess.run(['git','clone','--filter=blob:none','--no-checkout',src['url'],str(target)], check=True)
    subprocess.run(['git','-C',str(target),'fetch','--depth=1','origin',expected], check=True)
    subprocess.run(['git','-C',str(target),'checkout','--detach',expected], check=True)
    sha = subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'], text=True).strip()
    if sha != expected:
        sys.exit(f"Checkout mismatch for {src['name']}")
    print(f'PINNED {src["name"]}: {sha}', flush=True)
PY
