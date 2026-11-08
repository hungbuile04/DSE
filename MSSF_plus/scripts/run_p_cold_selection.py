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
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from MSSF_plus.protocol import get_fold_list
from MSSF_plus.train import run_experiment, _try_restore_completed_fold

P_COLD_CONFIGS = ['cand_A_p015', 'cand_A']
DIAGNOSTIC_REFS = ['MSSF_v2_selfcorr']


def _ensure_restored(exp_name: str, split_type: str, folds, results_dir: str, drive_dir: str):
    out_dir = os.path.join(results_dir, f"{exp_name}_{split_type}")
    drive_exp_dir = os.path.join(drive_dir, f"{exp_name}_{split_type}") if drive_dir else None
    all_ok = True
    for f in folds:
        fs = _try_restore_completed_fold(out_dir, f, drive_exp_dir=drive_exp_dir)
        if fs is None:
            all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description='Run cand_A with p_cold in {0.15, 0.25} on Folds 0-2, Split B (val_select)'
    )
    parser.add_argument('--splits', nargs='+', default=['B'], choices=['A', 'B'])
    parser.add_argument('--mode', type=str, default='dev', choices=['quick', 'dev', 'full'])
    parser.add_argument('--folds', type=int, nargs='+', default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--drive_dir', type=str, default=None)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--out_raw', type=str, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir or os.path.join(_PROJECT_ROOT, 'results_v2')
    folds = args.folds if args.folds is not None else get_fold_list(args.mode)

    for split_type in args.splits:
        for ref_name in DIAGNOSTIC_REFS:
            if not _ensure_restored(ref_name, split_type, folds, results_dir, args.drive_dir):
                run_experiment(
                    exp_name=ref_name,
                    split_type=split_type,
                    mode=args.mode,
                    custom_folds=args.folds,
                    device_str=args.device,
                    results_dir=results_dir,
                    drive_dir=args.drive_dir,
                    resume=(not args.no_resume),
                    verbose=True,
                )

        for cand_name in P_COLD_CONFIGS:
            run_experiment(
                exp_name=cand_name,
                split_type=split_type,
                mode=args.mode,
                custom_folds=args.folds,
                device_str=args.device,
                results_dir=results_dir,
                drive_dir=args.drive_dir,
                resume=(not args.no_resume),
                verbose=True,
            )

    out_raw = args.out_raw or os.path.join(
        _PROJECT_ROOT, 'reports', 'raw', 'p_cold_selection_summarize_results.txt'
    )
    methods_to_summarize = DIAGNOSTIC_REFS + P_COLD_CONFIGS
    cmd = [
        sys.executable, '-m', 'MSSF_plus.scripts.summarize_results',
        '--results_dir', results_dir,
        '--eval_target', 'valsel',
        '--ref_method', 'MSSF_v2_selfcorr',
        '--splits', *args.splits,
        '--folds', *[str(f) for f in folds],
        '--methods', *methods_to_summarize,
        '--out_raw', out_raw,
    ]
    if args.drive_dir:
        cmd.extend(['--drive_dir', args.drive_dir])

    print('\n' + '=' * 100)
    print('Running unified evaluation summary for p_cold in {0.15, 0.25} on cand_A...')
    print('=' * 100)
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
