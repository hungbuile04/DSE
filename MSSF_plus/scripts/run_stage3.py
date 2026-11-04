"""
Stage 3 Reference Benchmark Runner:
  - Runs `MSSF_v2_selfcorr` and `C3_ce_unweighted_ridge_selfcorr` (`profile_self_correction=1`,
    `c1_impute='regress_ridge'`) on Folds 0-2 for both Split B and Split A (`resume=True`, `eval_test=False`).
  - Invokes `MSSF_plus.scripts.summarize_results` to generate the verbatim Markdown output
    and saves it to `reports/raw/stage3_summarize_results.txt` (and backs up to Drive).
"""

import os
import sys
import shutil
import argparse
import subprocess

from MSSF_plus.protocol import DEFAULT_RESULTS_V2_DIR, get_device
from MSSF_plus.train import run_experiment


STAGE3_METHODS = [
    'MSSF_v2_selfcorr',
    'C3_ce_unweighted_ridge_selfcorr',
]


def main():
    parser = argparse.ArgumentParser(description="Run Stage 3 reference benchmarks (Folds 0-2, Split B & A) and save verbatim summarize_results output.")
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_V2_DIR)
    parser.add_argument('--drive_dir', type=str, default='/content/drive/MyDrive/DSE_results_v2')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--out_raw', type=str, default='reports/raw/stage3_summarize_results.txt')
    args = parser.parse_args()

    dev = get_device(args.device)
    drive_dir = args.drive_dir if (args.drive_dir and os.path.isdir(os.path.dirname(args.drive_dir))) else None

    for split_type in ['B', 'A']:
        for exp_name in STAGE3_METHODS:
            run_experiment(
                exp_name=exp_name,
                split_type=split_type,
                mode='dev',
                custom_folds=[0, 1, 2],
                results_dir=args.results_dir,
                drive_dir=drive_dir,
                resume=True,
                cli_overrides={'device': str(dev), 'eval_test': False},
                verbose=True,
            )

    cmd = [
        sys.executable, '-m', 'MSSF_plus.scripts.summarize_results',
        '--results_dir', args.results_dir,
        '--eval_target', 'valsel',
        '--splits', 'B', 'A',
        '--folds', '0', '1', '2',
        '--methods', *STAGE3_METHODS,
        '--ref_method', 'MSSF_v2_selfcorr',
        '--out_raw', args.out_raw,
    ]
    if drive_dir:
        cmd.extend(['--drive_dir', drive_dir])

    print("\n" + "=" * 110)
    print("=== STAGE 3 VERBATIM OUTPUT OF summarize_results.py ===")
    print("=" * 110)
    subprocess.run(cmd, check=True)

    if drive_dir and os.path.exists(args.out_raw):
        raw_drive_dir = os.path.join(drive_dir, 'raw')
        os.makedirs(raw_drive_dir, exist_ok=True)
        shutil.copy2(args.out_raw, os.path.join(raw_drive_dir, os.path.basename(args.out_raw)))


if __name__ == '__main__':
    main()
