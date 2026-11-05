"""
Unified CLI dispatcher for rerunning any baseline (or environment group of baselines)
under Protocol v2 (`load_splits_v2` -> `results_v2/`).

Examples:
  # Run pure-PyTorch baselines (Notebook 1 group: MGPred, SDPred, MSSF_v2) on Split B
  python -m MSSF_plus.baselines_rerun.run_all --baselines MGPred SDPred MSSF_v2 --splits B --mode full --resume

  # Run PyG + BPE baselines (Notebook 2 group: DSGAT, A3Net, HSTrans) on Split A and B
  python -m MSSF_plus.baselines_rerun.run_all --baselines DSGAT A3Net HSTrans --splits A B --mode full --resume

  # Run DGL baseline (Notebook 3 group: MVSL-DSF) on Split A and B
  python -m MSSF_plus.baselines_rerun.run_all --baselines MVSL-DSF --splits A B --mode full --resume
"""

import argparse
from typing import List, Optional

from MSSF_plus.protocol import DEFAULT_RESULTS_V2_DIR

ALL_BASELINES = [
    'MGPred',
    'MGPred_selfcorr',
    'SDPred',
    'SDPred_selfcorr',
    'MSSF_v2',
    'MSSF_v2_selfcorr',
    'DSGAT',
    'A3Net',
    'HSTrans',
    'MVSL-DSF',
    'MVSL-DSF_selfcorr',
    'CrossFeat',
    'CrossFeat_selfcorr',
]


def dispatch_baseline(
    baseline: str,
    split_type: str,
    mode: str = 'full',
    custom_folds: Optional[List[int]] = None,
    results_dir: str = DEFAULT_RESULTS_V2_DIR,
    drive_dir: Optional[str] = None,
    resume: bool = False,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    profile_self_correction: Optional[bool] = None,
    eval_test: bool = False,
    device: Optional[str] = None,
):
    if baseline.endswith('_selfcorr'):
        base_model = baseline[:-len('_selfcorr')]
        eff_self_corr = True
    else:
        base_model = baseline
        eff_self_corr = profile_self_correction

    overrides = {
        'epochs': epochs,
        'batch_size': batch_size,
        'profile_self_correction': eff_self_corr,
        'eval_test': eval_test,
        'device': device,
    }

    if base_model == 'MGPred':
        from MSSF_plus.baselines_rerun.run_mgpred import run_mgpred_experiment
        return run_mgpred_experiment(
            split_type=split_type, mode=mode, custom_folds=custom_folds,
            results_dir=results_dir, drive_dir=drive_dir, resume=resume, overrides=overrides,
        )
    if base_model == 'SDPred':
        from MSSF_plus.baselines_rerun.run_sdpred import run_sdpred_experiment
        return run_sdpred_experiment(
            split_type=split_type, mode=mode, custom_folds=custom_folds,
            results_dir=results_dir, drive_dir=drive_dir, resume=resume, overrides=overrides,
        )
    if base_model == 'MSSF_v2':
        from MSSF_plus.train import run_experiment
        clean_overrides = {
            k: v for k, v in {
                'epochs': epochs,
                'batch_size': batch_size,
                'profile_self_correction': eff_self_corr,
                'eval_test': eval_test,
                'device': device,
            }.items() if v is not None
        }
        exp_name = 'MSSF_v2_selfcorr' if eff_self_corr else 'MSSF_v2'
        return run_experiment(
            exp_name=exp_name, split_type=split_type, mode=mode, custom_folds=custom_folds,
            results_dir=results_dir, drive_dir=drive_dir, resume=resume,
            cli_overrides=clean_overrides,
        )
    if base_model in ('DSGAT', 'A3Net'):
        from MSSF_plus.baselines_rerun.run_dsgat_a3net import run_pyg_baseline_experiment
        pyg_overrides = {'epochs': epochs, 'train_batch': batch_size, 'eval_test': eval_test, 'device': device}
        return run_pyg_baseline_experiment(
            model_name=base_model, split_type=split_type, mode=mode, custom_folds=custom_folds,
            results_dir=results_dir, drive_dir=drive_dir, resume=resume, overrides=pyg_overrides,
        )
    if base_model == 'HSTrans':
        from MSSF_plus.baselines_rerun.run_hstrans import run_hstrans_experiment
        return run_hstrans_experiment(
            split_type=split_type, mode=mode, custom_folds=custom_folds,
            results_dir=results_dir, drive_dir=drive_dir, resume=resume, overrides=overrides,
        )
    if base_model in ('MVSL-DSF', 'CrossFeat'):
        from MSSF_plus.baselines_rerun.run_mvsl import run_mvsl_experiment
        return run_mvsl_experiment(
            split_type=split_type, mode=mode, custom_folds=custom_folds,
            results_dir=results_dir, drive_dir=drive_dir, resume=resume, overrides=overrides,
        )
    raise ValueError(f"Unknown baseline {baseline!r}. Expected one of {ALL_BASELINES}")


def main():
    parser = argparse.ArgumentParser(description='Unified Protocol v2 baseline dispatcher.')
    parser.add_argument(
        '--baselines', '--models', dest='baselines', type=str, nargs='+', required=True, choices=ALL_BASELINES,
        help='One or more baselines to run',
    )
    parser.add_argument('--splits', type=str, nargs='+', default=['B'], choices=['A', 'B'])
    parser.add_argument('--mode', type=str, default='full', choices=['quick', 'dev', 'full'])
    parser.add_argument('--folds', type=int, nargs='*', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--profile_self_correction', type=int, default=None, choices=[0, 1])
    parser.add_argument('--eval_test', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_V2_DIR)
    parser.add_argument('--drive_dir', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    self_corr_flag = (
        bool(args.profile_self_correction) if args.profile_self_correction is not None else None
    )

    for split_type in args.splits:
        for baseline in args.baselines:
            dispatch_baseline(
                baseline=baseline,
                split_type=split_type,
                mode=args.mode,
                custom_folds=args.folds,
                results_dir=args.results_dir,
                drive_dir=args.drive_dir,
                resume=args.resume,
                epochs=args.epochs,
                batch_size=args.batch_size,
                profile_self_correction=self_corr_flag,
                eval_test=bool(args.eval_test),
                device=args.device,
            )


if __name__ == '__main__':
    main()
