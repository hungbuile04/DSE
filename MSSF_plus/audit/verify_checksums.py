"""SHA256 checksum recorder and verifier for shared_data/ (Invariant Principle #1).

Usage:
  python verify_checksums.py --init    # Save baseline SHA256 of all existing files in shared_data/
  python verify_checksums.py --check   # Verify all baseline files are unchanged; list newly added files
"""
import os
import sys
import json
import hashlib
import argparse

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(AUDIT_DIR))
SHARED_DIR = os.path.join(REPO_DIR, 'shared_data')
BASELINE_JSON = os.path.join(AUDIT_DIR, 'outputs', 'shared_data_sha256_baseline.json')

IGNORE_NAMES = {'.DS_Store'}
IGNORE_DIRS = {'__pycache__'}
# Step 2a explicitly authorizes updating shared_data/README.md to document splits_val and fix 2 doc typos
DOC_FILES_ALLOWED_TO_UPDATE = {'README.md'}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def scan_shared_data():
    manifest = {}
    for root, dirs, files in os.walk(SHARED_DIR):
        dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS)
        for fname in sorted(files):
            if fname in IGNORE_NAMES:
                continue
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, SHARED_DIR)
            manifest[rel_path] = {
                'sha256': sha256_file(full_path),
                'size_bytes': os.path.getsize(full_path),
            }
    return manifest


def init_baseline():
    os.makedirs(os.path.dirname(BASELINE_JSON), exist_ok=True)
    if os.path.exists(BASELINE_JSON):
        print(f'[ABORT] Baseline file already exists at {BASELINE_JSON}. Use --check instead.')
        sys.exit(1)
    manifest = scan_shared_data()
    with open(BASELINE_JSON, 'w') as f:
        json.dump({'n_files': len(manifest), 'files': manifest}, f, indent=2)
    print(f'[OK] Recorded baseline SHA256 for {len(manifest)} files in shared_data/ -> {BASELINE_JSON}')


def check_baseline():
    if not os.path.exists(BASELINE_JSON):
        print(f'[ERROR] Baseline file not found: {BASELINE_JSON}')
        sys.exit(1)
    with open(BASELINE_JSON, 'r') as f:
        base = json.load(f)['files']
    cur = scan_shared_data()

    missing = []
    modified_unauthorized = []
    modified_doc = []
    unchanged_count = 0

    for rel, info in base.items():
        if rel not in cur:
            missing.append(rel)
        elif cur[rel]['sha256'] != info['sha256']:
            if rel in DOC_FILES_ALLOWED_TO_UPDATE:
                modified_doc.append((rel, info['sha256'], cur[rel]['sha256']))
            else:
                modified_unauthorized.append((rel, info['sha256'], cur[rel]['sha256']))
        else:
            unchanged_count += 1

    added = sorted(rel for rel in cur if rel not in base)

    print('=== shared_data/ SHA256 Integrity Check ===')
    print(f'  Baseline files tracked : {len(base)}')
    print(f'  Bit-for-bit unchanged  : {unchanged_count}')
    print(f'  Authorized doc updates : {len(modified_doc)} {[x[0] for x in modified_doc]}')
    print(f'  Newly added files      : {len(added)}')
    if missing:
        print(f'  [FAIL] Missing files   : {missing}')
    if modified_unauthorized:
        print(f'  [FAIL] Modified files  : {modified_unauthorized}')

    if missing or modified_unauthorized:
        sys.exit(1)
    print('  [PASS] All original data, split, and code files in shared_data/ are 100% intact.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--init', action='store_true')
    group.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.init:
        init_baseline()
    else:
        check_baseline()
