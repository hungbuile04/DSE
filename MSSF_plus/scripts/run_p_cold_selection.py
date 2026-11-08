"""
Run transparent hyperparameter selection for `p_cold` (`c1_view_dropout_prob` in {0.15, 0.25})
on the selected final model `cand_A` across Folds 0, 1, 2 on Split B (`val_select`),
then generate the verbatim summary report via `MSSF_plus/scripts/summarize_results.py`.

Usage (Colab):
  python3 -m MSSF_plus.scripts.run_p_cold_selection \
      --splits B --mode dev --device cuda \
      --results_dir /content/DSE/results_v2 \
      --drive_dir /content/drive/MyDrive/DSE_results_v2
"""

import argparse
import os
import shutil
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from MSSF_plus.protocol import REPO_DIR, DEFAULT_RESULTS_V2_DIR
from MSSF_plus.train import run_experiment

P_COLD_EXPS = ['MSSF_v2_selfcorr', 'cand_A', 'cand_A_p015']


def main():
    parser = argparse.ArgumentParser(
        description='Run cand_A with p_cold in {0.15, 0.25} on Folds 0-2, Split B (val_select)'
    )
    parser.add_argument('--splits', nargs='+', default=['B'], choices=['A', 'B'])
    parser.add_argument('--mode', type=str, default='dev', choices=['quick', 'dev', 'full'])
    parser.add_argument('--folds', type=int, nargs='*', default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_V2_DIR)
    parser.add_argument('--drive_dir', type=str, default=None)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--summarize_only', action='store_true')
    parser.add_argument('--out_raw', type=str, default=None)
    args = parser.parse_args()

    folds = (
        args.folds
        if args.folds is not None
        else ([0] if args.mode == 'quick' else ([0, 1, 2] if args.mode == 'dev' else list(range(10))))
    )
    overrides = {'epochs': int(args.epochs)}
    if args.device is not None:
        overrides['device'] = args.device

    if not args.summarize_only:
        for split_type in args.splits:
            for exp_name in P_COLD_EXPS:
                run_experiment(
                    exp_name=exp_name,
                    split_type=split_type,
                    mode=args.mode,
                    custom_folds=args.folds,
                    results_dir=args.results_dir,
                    drive_dir=args.drive_dir,
                    resume=(not args.no_resume),
                    cli_overrides=overrides,
                    verbose=True,
                )

    raw_dir = os.path.join(REPO_DIR, 'reports', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    out_raw = args.out_raw or os.path.join(raw_dir, 'p_cold_selection_summarize_results.txt')

    cmd = [
        sys.executable, '-m', 'MSSF_plus.scripts.summarize_results',
        '--results_dir', args.results_dir,
        '--eval_target', 'valsel',
        '--ref_method', 'MSSF_v2_selfcorr',
        '--splits', *args.splits,
        '--folds', *[str(f) for f in folds],
        '--methods', *P_COLD_EXPS,
        '--out_raw', out_raw,
    ]
    if args.drive_dir:
        cmd.extend(['--drive_dir', args.drive_dir])

    print('\n' + '=' * 100)
    print('Running unified evaluation summary for p_cold in {0.15, 0.25} on cand_A...')
    print('=' * 100)
    subprocess.run(cmd, check=True)

    if args.drive_dir and os.path.exists(out_raw):
        raw_drive_dir = os.path.join(args.drive_dir, 'reports_raw')
        os.makedirs(raw_drive_dir, exist_ok=True)
        shutil.copy2(out_raw, os.path.join(raw_drive_dir, os.path.basename(out_raw)))


if __name__ == '__main__':
    main()
