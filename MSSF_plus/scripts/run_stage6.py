"""
Stage 6 — External Baselines Runner & Verbatim Report Generator (Folds 0-2, Split B).

Runs external baselines under Protocol v2 on Folds 0-2, Split B (`val_select` with `eval_test=False`):
  - Priority baselines:
      * DSGAT (+ DSGAT_thr)
      * A3Net (+ A3Net_thr)
      * HSTrans (+ HSTrans_thr)
      * MGPred & MGPred_selfcorr (+ MGPred_thr, MGPred_selfcorr_thr)
      * SDPred & SDPred_selfcorr (+ SDPred_thr, SDPred_selfcorr_thr)
      * MVSL-DSF & MVSL-DSF_selfcorr [CrossFeat] (+ MVSL-DSF_thr, MVSL-DSF_selfcorr_thr)
  - Automatically generates `_thr` (4-threshold calibrated on `val_select`) variants for all regression baselines.
  - Compares against Stage 3 approved references (`MSSF_v2_selfcorr` and `C3_ce_unweighted_ridge_selfcorr`)
    using verbatim output of `MSSF_plus.scripts.summarize_results`, saved to
    `reports/raw/stage6_baselines_summarize_results.txt`.
"""

import os
import sys
import shutil
import argparse
import subprocess
from typing import List

from MSSF_plus.protocol import REPO_DIR, DEFAULT_RESULTS_V2_DIR
from MSSF_plus.baselines_rerun.run_all import dispatch_baseline

DEFAULT_STAGE6_BASELINES = [
    'DSGAT',
    'A3Net',
    'HSTrans',
    'MGPred',
    'MGPred_selfcorr',
    'SDPred',
    'SDPred_selfcorr',
    'MVSL-DSF',
    'MVSL-DSF_selfcorr',
]

SUMMARY_METHODS_STAGE6 = [
    'MSSF_v2_selfcorr',
    'C3_ce_unweighted_ridge_selfcorr',
    'DSGAT',
    'DSGAT_thr',
    'A3Net',
    'A3Net_thr',
    'HSTrans',
    'HSTrans_thr',
    'MGPred',
    'MGPred_thr',
    'MGPred_selfcorr',
    'MGPred_selfcorr_thr',
    'SDPred',
    'SDPred_thr',
    'SDPred_selfcorr',
    'SDPred_selfcorr_thr',
    'MVSL-DSF',
    'MVSL-DSF_thr',
    'MVSL-DSF_selfcorr',
    'MVSL-DSF_selfcorr_thr',
]


def main():
    parser = argparse.ArgumentParser(description='Run Stage 6 baselines and save verbatim summarize_results output.')
    parser.add_argument(
        '--baselines',
        type=str,
        nargs='+',
        default=DEFAULT_STAGE6_BASELINES,
        help='Baselines to run (default: DSGAT A3Net HSTrans MGPred MGPred_selfcorr SDPred SDPred_selfcorr MVSL-DSF MVSL-DSF_selfcorr)',
    )
    parser.add_argument('--splits', type=str, nargs='+', default=['B'], choices=['A', 'B'])
    parser.add_argument('--mode', type=str, default='dev', choices=['quick', 'dev', 'full'])
    parser.add_argument('--folds', type=int, nargs='*', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_V2_DIR)
    parser.add_argument('--drive_dir', type=str, default=None)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--summarize_only', action='store_true')
    args = parser.parse_args()

    folds = args.folds if args.folds is not None else ([0] if args.mode == 'quick' else ([0, 1, 2] if args.mode == 'dev' else list(range(10))))

    if not args.summarize_only:
        for split_type in args.splits:
            for baseline in args.baselines:
                try:
                    dispatch_baseline(
                        baseline=baseline,
                        split_type=split_type,
                        mode=args.mode,
                        custom_folds=args.folds,
                        results_dir=args.results_dir,
                        drive_dir=args.drive_dir,
                        resume=True,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        eval_test=False,
                        device=args.device,
                    )
                except ImportError as exc:
                    print(
                        f"  [SKIP-ENV] Skipping {baseline} in this environment due to missing optional dependency: {exc}"
                    )

    raw_dir = os.path.join(REPO_DIR, 'reports', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    out_raw = os.path.join(raw_dir, 'stage6_baselines_summarize_results.txt')

    cmd = [
        sys.executable,
        '-m',
        'MSSF_plus.scripts.summarize_results',
        '--results_dir',
        args.results_dir,
        '--eval_target',
        'valsel',
        '--ref_method',
        'MSSF_v2_selfcorr',
        '--splits',
        *args.splits,
        '--folds',
        *[str(f) for f in folds],
        '--methods',
        *SUMMARY_METHODS_STAGE6,
        '--out_raw',
        out_raw,
    ]
    if args.drive_dir:
        cmd.extend(['--drive_dir', args.drive_dir])
    subprocess.run(cmd, check=True)

    if args.drive_dir and os.path.exists(out_raw):
        raw_drive_dir = os.path.join(args.drive_dir, 'reports_raw')
        os.makedirs(raw_drive_dir, exist_ok=True)
        shutil.copy2(out_raw, os.path.join(raw_drive_dir, os.path.basename(out_raw)))


if __name__ == '__main__':
    main()
