"""Read-only, dependency-free prepublication checks. Never prints secret values."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {'.urdf', '.usd', '.usda', '.usdc', '.stl', '.obj', '.dae', '.mtl',
             '.pt', '.pth', '.onnx', '.jit', '.npz', '.npy', '.pkl', '.pickle',
             '.mp4', '.mov', '.avi', '.webm', '.png', '.jpg', '.jpeg', '.gif',
             '.bundle', '.zip', '.gz', '.tgz', '.tar', '.partial', '.pem', '.key'}
PATTERNS = {
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'aws_access_key': re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'credential_url': re.compile(r'https?://[^\s/"\x27]+:[^\s/@"\x27]+@'),
    'private_machine_path': re.compile(r'/(?:Users/[^/\s]+|home/(?:alex|tongji)|hy-tmp/chao5090)/'),
}


def files():
    return sorted(p for p in ROOT.rglob('*') if '.git' not in p.relative_to(ROOT).parts
                  and '__pycache__' not in p.parts and p.name != '.DS_Store' and p.is_file())


def audit(history=False):
    errors = []
    inventory = []
    python_count = 0
    for p in files():
        rel = p.relative_to(ROOT).as_posix()
        if p.is_symlink():
            errors.append([rel, 'symlink'])
        if p.suffix.lower() in FORBIDDEN or p.name.startswith('.env') or p.name == '.gitmodules':
            errors.append([rel, 'forbidden publication type'])
        raw = p.read_bytes()
        if len(raw) > 1_100_000:
            errors.append([rel, 'file exceeds 1.1 MB review cap'])
        try:
            data = raw.decode('utf-8')
        except UnicodeDecodeError:
            errors.append([rel, 'non-UTF8/binary payload'])
            continue
        if '\x00' in data:
            errors.append([rel, 'binary NUL'])
        for label, pattern in PATTERNS.items():
            if pattern.search(data):
                errors.append([rel, label])
        if p.suffix == '.py':
            python_count += 1
            try:
                ast.parse(data, filename=rel)
            except SyntaxError:
                errors.append([rel, 'Python syntax'])
        if p.suffix == '.json':
            try:
                json.loads(data)
            except json.JSONDecodeError:
                errors.append([rel, 'JSON syntax'])
        inventory.append(dict(path=rel, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    for item in json.loads((ROOT/'provenance/files.json').read_text()):
        p = ROOT/item['destination']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item['published_sha256']:
            errors.append([item['destination'], 'provenance mismatch'])
    release_manifest = ROOT/'provenance/release_files.json'
    if release_manifest.is_file():
        declared = json.loads(release_manifest.read_text())['files']
        actual = {v['path']: v for v in inventory if v['path'] != 'provenance/release_files.json'}
        if {v['path'] for v in declared} != set(actual):
            errors.append(['provenance/release_files.json', 'release file set mismatch'])
        for item in declared:
            if actual.get(item['path']) != item:
                errors.append([item['path'], 'release file hash/size mismatch'])
    git = None
    if history:
        def call(*args):
            return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
        roots = call('rev-list', '--max-parents=0', 'HEAD').splitlines()
        git = dict(commit_count=int(call('rev-list', '--count', 'HEAD')), root_count=len(roots),
                   root_has_no_parent=len(call('rev-list', '--parents', '-n', '1', roots[0]).split()) == 1,
                   branch=call('branch', '--show-current'),
                   shallow=call('rev-parse', '--is-shallow-repository'),
                   alternates=(ROOT/'.git/objects/info/alternates').exists())
        if len(roots) != 1 or not git['root_has_no_parent'] or git['alternates'] or git['shallow'] != 'false':
            errors.append(['.git', 'fresh-history check failed'])
    return dict(passed=not errors, file_count=len(inventory), total_bytes=sum(x['bytes'] for x in inventory),
                python_files=python_count, errors=errors, git=git, files=inventory)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history', action='store_true')
    p.add_argument('--manifest', type=Path, help='Optional output outside the release tree')
    args = p.parse_args()
    result = audit(args.history)
    if args.manifest:
        args.manifest.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
