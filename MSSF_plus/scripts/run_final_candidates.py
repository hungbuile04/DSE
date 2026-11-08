"""
Runner for Post-Stage 4 Final Model Candidates (`cand_A`, `cand_B`, `cand_C`, `cand_D`)
on Folds 0-2, Splits B and A, with `profile_self_correction = 1`:

  - `cand_A`: vdrop_only + C2 + C5 + CE, drop BVI (`use_bvi=False`), NO imputation (`c1_impute='zero'`), NO modulator (`c1_use_availability=False`)
  - `cand_B`: like `cand_A` but keep modulator (`c1_use_availability=True`)
  - `cand_C`: like `cand_A` but add ridge imputation (`c1_impute='regress_ridge'`)
  - `cand_D`: `= full_no_bvi` (keep modulator + ridge imputation, for reference)

Also includes diagnostic BVI configurations (`C1_avail_only`, `loo_no_C1`, `C3_ce_unweighted_ridge_selfcorr`)
in the verbatim `summarize_results.py` report to preserve the posterior uncertainty `sigma` comparison
(missing `D9, D10` has higher `sigma` without imputation and equalizes with `regress_ridge` imputation).
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from typing import List, Optional

from MSSF_plus.protocol import REPO_DIR, DEFAULT_RESULTS_V2_DIR
from MSSF_plus.configs.experiments import get_experiment_config
from MSSF_plus.train import run_experiment, _try_restore_completed_fold

CANDIDATE_EXPS = ['cand_A', 'cand_B', 'cand_C', 'cand_D']
DIAGNOSTIC_REF_EXPS = [
    'MSSF_v2_selfcorr',
    'cand_A',
    'cand_B',
    'cand_C',
    'cand_D',
    'C1_avail_only',
    'loo_no_C1',
    'C3_ce_unweighted_ridge_selfcorr',
]


def _seed_cand_d_from_full_no_bvi(
    split_type: str,
    folds: List[int],
    results_dir: str,
    drive_dir: Optional[str],
) -> None:
    """
    Since `cand_D` is defined identically to `full_no_bvi`, reuse existing `full_no_bvi_{split}`
    fold artifacts if they are already completed in `results_dir` or `drive_dir`.
    """
    src_folder = f"full_no_bvi_{split_type}"
    dst_folder = f"cand_D_{split_type}"
    src_local = os.path.join(results_dir, src_folder)
    dst_local = os.path.join(results_dir, dst_folder)
    dst_drive = os.path.join(drive_dir, dst_folder) if drive_dir else None

    cfg_cand_d = get_experiment_config('cand_D')
    cfg_full_nb = get_experiment_config('full_no_bvi')

    for fold in folds:
        if _try_restore_completed_fold(
            dst_local, drive_dir, dst_folder, fold, eval_test=False, cfg=cfg_cand_d, split_type=split_type
        ) is not None:
            continue
        if _try_restore_completed_fold(
            src_local, drive_dir, src_folder, fold, eval_test=False, cfg=cfg_full_nb, split_type=split_type
        ) is None:
            continue

        os.makedirs(dst_local, exist_ok=True)
        if dst_drive:
            os.makedirs(dst_drive, exist_ok=True)

        prefix = f"fold_{fold}_"
        for fname in os.listdir(src_local):
            if not fname.startswith(prefix):
                continue
            s_path = os.path.join(src_local, fname)
            d_path = os.path.join(dst_local, fname)
            if fname.endswith('.json'):
                with open(s_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                meta['exp_name'] = 'cand_D'
                if isinstance(meta.get('config'), dict):
                    meta['config']['exp_name'] = 'cand_D'
                with open(d_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=2)
            else:
                shutil.copy2(s_path, d_path)
            if dst_drive:
                shutil.copy2(d_path, os.path.join(dst_drive, fname))


def _run_summarize_verbatim(
    results_dir: str,
    drive_dir: Optional[str],
    splits: List[str],
    folds: List[int],
    methods: List[str],
    out_raw: str,
) -> None:
    cmd = [
        sys.executable,
        '-m',
        'MSSF_plus.scripts.summarize_results',
        '--results_dir',
        results_dir,
        '--eval_target',
        'valsel',
        '--ref_method',
        'MSSF_v2_selfcorr',
        '--splits',
        *splits,
        '--folds',
        *[str(f) for f in folds],
        '--methods',
        *methods,
        '--out_raw',
        out_raw,
    ]
    if drive_dir:
        cmd.extend(['--drive_dir', drive_dir])
    subprocess.run(cmd, check=True)
    if drive_dir and os.path.exists(out_raw):
        raw_drive_dir = os.path.join(drive_dir, 'reports_raw')
        os.makedirs(raw_drive_dir, exist_ok=True)
        shutil.copy2(out_raw, os.path.join(raw_drive_dir, os.path.basename(out_raw)))


def main():
    parser = argparse.ArgumentParser(
        description='Run final model candidates (cand_A..cand_D) on Folds 0-2, Splits B & A, and output verbatim summary.'
    )
    parser.add_argument('--splits', type=str, nargs='+', default=['B', 'A'], choices=['A', 'B'])
    parser.add_argument('--mode', type=str, default='dev', choices=['quick', 'dev', 'full'])
    parser.add_argument('--folds', type=int, nargs='*', default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_V2_DIR)
    parser.add_argument('--drive_dir', type=str, default=None)
    parser.add_argument('--summarize_only', action='store_true')
    args = parser.parse_args()

    folds = (
        args.folds
        if args.folds is not None
        else ([0] if args.mode == 'quick' else ([0, 1, 2] if args.mode == 'dev' else list(range(10))))
    )
    overrides = {'epochs': int(args.epochs)}
    if args.device is not None:
        overrides['device'] = args.device

    for split_type in args.splits:
        _seed_cand_d_from_full_no_bvi(split_type, folds, args.results_dir, args.drive_dir)

    if not args.summarize_only:
        for split_type in args.splits:
            for exp_name in CANDIDATE_EXPS:
                run_experiment(
                    exp_name=exp_name,
                    split_type=split_type,
                    mode=args.mode,
                    custom_folds=args.folds,
                    results_dir=args.results_dir,
                    drive_dir=args.drive_dir,
                    resume=True,
                    cli_overrides=overrides,
                    verbose=True,
                )

    raw_dir = os.path.join(REPO_DIR, 'reports', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    out_raw = os.path.join(raw_dir, 'final_candidates_summarize_results.txt')
    _run_summarize_verbatim(
        results_dir=args.results_dir,
        drive_dir=args.drive_dir,
        splits=args.splits,
        folds=folds,
        methods=DIAGNOSTIC_REF_EXPS,
        out_raw=out_raw,
    )


if __name__ == '__main__':
    main()
