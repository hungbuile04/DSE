"""
Stage 4 — Full Component & Leave-One-Out Ablation Runner & Verbatim Report Generator.

Runs Stage 4 ablation groups on Folds 0-2 (mode='dev'), Splits B and A, with:
  - profile_self_correction = True
  - C1 default imputer = 'regress_ridge'
  - C1 group: MSSF_v2_selfcorr, C1_avail_only, C1_vdrop_only, C1_avail_vdrop_zero,
              C1_avail_vdrop_drop, C1_avail_vdrop_mean, C1_avail_vdrop_knn,
              C1_avail_vdrop_ridge, C1_avail_vdrop_regress, C1_ridge_no_modulator
  - C2 group: MSSF_v2_selfcorr, C2_plus_ontology, C2_no_profile
  - C3 group: C3_ce_unweighted_ridge_selfcorr, C3_ce_weighted, C3_emd_unweighted,
              C3_emd_weighted, C3_corn_unweighted, C3_corn_weighted
  - C5 group: MSSF_v2_selfcorr, C5_knn, C5_single_view, C5_multi_view,
              C5_multi_view_aux, C5_multi_view_fixedK6 (with *_k24 removed per Stage 3 verification)
  - BVI 2x2 factorial group: C3_ce_unweighted_ridge_selfcorr (keep BVI, C5 on),
                             full_no_bvi (drop BVI, C5 on),
                             bvi_keep_no_c5 (keep BVI, C5 off),
                             full_no_bvi_no_c5 (drop BVI, C5 off)
  - LOO group: MSSF_v2_selfcorr, C3_ce_unweighted_ridge_selfcorr, MSSF_plus_full,
               loo_no_C1, loo_no_C2, loo_no_C3, loo_no_C5

After running the requested group(s), invokes `MSSF_plus.scripts.summarize_results`
and saves verbatim raw outputs to `reports/raw/stage4_*_summarize_results.txt`.
"""

import os
import sys
import shutil
import argparse
import subprocess
from typing import List

from MSSF_plus.protocol import REPO_DIR, DEFAULT_RESULTS_V2_DIR
from MSSF_plus.scripts.run_ablation import ABLATION_GROUPS, get_experiments_for_group
from MSSF_plus.train import run_experiment

STAGE4_GROUPS = ['c1', 'c2', 'c3', 'c5', 'bvi', 'loo']


def _run_summarize_verbatim(
    results_dir: str,
    drive_dir: str,
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
    parser = argparse.ArgumentParser(description='Run Stage 4 ablations and save verbatim summarize_results outputs.')
    parser.add_argument(
        '--groups',
        type=str,
        nargs='+',
        default=['all'],
        choices=STAGE4_GROUPS + ['all'],
        help='Stage 4 ablation group(s) to run (default: all)',
    )
    parser.add_argument('--splits', type=str, nargs='+', default=['B', 'A'], choices=['A', 'B'])
    parser.add_argument('--mode', type=str, default='dev', choices=['quick', 'dev', 'full'])
    parser.add_argument('--folds', type=int, nargs='*', default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_V2_DIR)
    parser.add_argument('--drive_dir', type=str, default=None)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--summarize_only', action='store_true', help='Only generate verbatim reports from existing results')
    args = parser.parse_args()

    active_groups = STAGE4_GROUPS if 'all' in args.groups else list(args.groups)
    folds = args.folds if args.folds is not None else ([0] if args.mode == 'quick' else ([0, 1, 2] if args.mode == 'dev' else list(range(10))))

    overrides = {'epochs': int(args.epochs)}
    if args.device is not None:
        overrides['device'] = args.device

    if not args.summarize_only:
        # Deduplicate experiments in execution order across requested groups
        ordered_exps = []
        for g in active_groups:
            for exp_name in ABLATION_GROUPS[g]:
                if exp_name not in ordered_exps:
                    ordered_exps.append(exp_name)

        for split_type in args.splits:
            for exp_name in ordered_exps:
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

    for g in active_groups:
        out_raw_g = os.path.join(raw_dir, f'stage4_{g}_summarize_results.txt')
        _run_summarize_verbatim(
            results_dir=args.results_dir,
            drive_dir=args.drive_dir,
            splits=args.splits,
            folds=folds,
            methods=ABLATION_GROUPS[g],
            out_raw=out_raw_g,
        )

    if len(active_groups) > 1:
        all_methods = get_experiments_for_group('stage4', mode=args.mode)
        out_raw_all = os.path.join(raw_dir, 'stage4_all_summarize_results.txt')
        _run_summarize_verbatim(
            results_dir=args.results_dir,
            drive_dir=args.drive_dir,
            splits=args.splits,
            folds=folds,
            methods=all_methods,
            out_raw=out_raw_all,
        )


if __name__ == '__main__':
    main()
