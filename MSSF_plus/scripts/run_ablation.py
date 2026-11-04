"""
Batch ablation runner for MSSF_plus experiments.

Supports running predefined experiment groups ('baseline', 'c1', 'c2', 'c3', 'c5', 'loo', 'main', 'all')
under 'quick' (fold 0), 'dev' (folds 0,1,2), or 'full' (folds 0..9) mode.
"""

import argparse
from typing import Dict, List

from MSSF_plus.protocol import DEFAULT_RESULTS_V2_DIR
from MSSF_plus.train import run_experiment

ABLATION_GROUPS: Dict[str, List[str]] = {
    'baseline': ['MSSF_v2_selfcorr'],
    'stage3': [
        'MSSF_v2_selfcorr',
        'C3_ce_unweighted_ridge_selfcorr',
    ],
    'c1': [
        'MSSF_v2_selfcorr',
        'C1_avail_only',
        'C1_vdrop_only',
        'C1_avail_vdrop_zero',
        'C1_avail_vdrop_drop',
        'C1_avail_vdrop_mean',
        'C1_avail_vdrop_ridge',
        'C1_ridge_no_modulator',
        'C1_avail_vdrop_knn',
        'C1_avail_vdrop_regress',
    ],
    'c2': [
        'MSSF_v2_selfcorr',
        'C2_plus_ontology',
        'C2_no_profile',
    ],
    'c3': [
        'C3_ce_unweighted_ridge_selfcorr',
        'C3_ce_weighted',
        'C3_emd_unweighted',
        'C3_emd_weighted',
        'C3_corn_unweighted',
        'C3_corn_weighted',
    ],
    'c5': [
        'MSSF_v2_selfcorr',
        'C5_knn',
        'C5_single_view',
        'C5_multi_view',
        'C5_multi_view_aux',
        'C5_single_view_k24',
        'C5_multi_view_k24',
        'C5_multi_view_aux_k24',
    ],
    'loo': [
        'MSSF_v2_selfcorr',
        'C3_ce_unweighted_ridge_selfcorr',
        'MSSF_plus_full',
        'loo_no_C1',
        'loo_no_C2',
        'loo_no_C3',
        'loo_no_C5',
    ],
    'main': [
        'MSSF_v2_selfcorr',
        'C3_ce_unweighted_ridge_selfcorr',
        'MSSF_plus_full',
    ],
    'stage2': [
        'MSSF_v2',
        'MSSF_v2_selfcorr',
        'C3_ce_unweighted',
        'C3_ce_unweighted_selfcorr',
    ],
    'bvi': [
        'C3_ce_unweighted_ridge_selfcorr',
        'full_no_bvi',
    ],
}


def get_experiments_for_group(group: str, mode: str = 'dev') -> List[str]:
    if group in ('all', 'stage4'):
        seen = []
        for g in ['c1', 'c2', 'c3', 'c5', 'bvi', 'loo']:
            for exp in ABLATION_GROUPS[g]:
                if exp not in seen:
                    seen.append(exp)
        exps = seen
    elif group in ABLATION_GROUPS:
        exps = list(ABLATION_GROUPS[group])
    else:
        raise ValueError(f"Unknown group {group!r}; choose from {list(ABLATION_GROUPS.keys()) + ['stage4', 'all']}")

    # Diffusion is excluded from all runs per Stage 2 follow-up decision
    exps = [e for e in exps if e != 'C1_avail_vdrop_diffusion']
    return exps


def main():
    parser = argparse.ArgumentParser(description='Run MSSF_plus ablation suite.')
    parser.add_argument(
        '--group',
        type=str,
        default='main',
        choices=list(ABLATION_GROUPS.keys()) + ['stage4', 'all'],
        help='Ablation group to run',
    )
    parser.add_argument(
        '--exps',
        type=str,
        nargs='+',
        default=None,
        help='Optional explicit list of experiment names (overrides --group)',
    )
    parser.add_argument('--splits', type=str, nargs='+', default=['B'], choices=['A', 'B'])
    parser.add_argument('--mode', type=str, default='dev', choices=['quick', 'dev', 'full'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--imputer_epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_V2_DIR)
    parser.add_argument('--drive_dir', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--profile_self_correction', type=int, choices=[0, 1], default=None)
    parser.add_argument('--eval_test', action='store_true')
    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None:
        overrides['epochs'] = args.epochs
    if args.imputer_epochs is not None:
        overrides['c1_imputer_epochs'] = args.imputer_epochs
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.device is not None:
        overrides['device'] = args.device
    if args.profile_self_correction is not None:
        overrides['profile_self_correction'] = bool(args.profile_self_correction)
    if args.eval_test:
        overrides['eval_test'] = True

    if args.exps:
        exps = list(args.exps)
    else:
        exps = get_experiments_for_group(args.group, mode=args.mode)
    for split_type in args.splits:
        for exp_name in exps:
            run_experiment(
                exp_name=exp_name,
                split_type=split_type,
                mode=args.mode,
                results_dir=args.results_dir,
                drive_dir=args.drive_dir,
                resume=args.resume,
                cli_overrides=overrides,
                verbose=True,
            )


if __name__ == '__main__':
    main()
