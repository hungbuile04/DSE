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
    'baseline': ['MSSF_v2'],
    'c1': [
        'MSSF_v2',
        'C1_avail_only',
        'C1_vdrop_only',
        'C1_avail_vdrop_zero',
        'C1_avail_vdrop_drop',
        'C1_avail_vdrop_mean',
        'C1_avail_vdrop_regress',
        'C1_avail_vdrop_diffusion',
    ],
    'c2': [
        'MSSF_v2',
        'C2_plus_ontology',
        'C2_no_profile',
    ],
    'c3': [
        'C3_ce_unweighted',
        'C3_ce_weighted',
        'C3_emd_unweighted',
        'C3_emd_weighted',
        'C3_corn_unweighted',
        'C3_corn_weighted',
        'C3_softquant_unweighted',
        'C3_softquant_weighted',
    ],
    'c5': [
        'MSSF_v2',
        'C5_knn',
        'C5_single_view',
        'C5_multi_view',
        'C5_multi_view_aux',
    ],
    'loo': [
        'MSSF_v2',
        'MSSF_plus_full',
        'loo_no_C1',
        'loo_no_C2',
        'loo_no_C3',
        'loo_no_C5',
    ],
    'main': [
        'MSSF_v2',
        'MSSF_plus_full',
    ],
}


def get_experiments_for_group(group: str) -> List[str]:
    if group == 'all':
        seen = []
        for g in ['baseline', 'c1', 'c2', 'c3', 'c5', 'loo']:
            for exp in ABLATION_GROUPS[g]:
                if exp not in seen:
                    seen.append(exp)
        return seen
    if group not in ABLATION_GROUPS:
        raise ValueError(f"Unknown group {group!r}; choose from {list(ABLATION_GROUPS.keys()) + ['all']}")
    return ABLATION_GROUPS[group]


def main():
    parser = argparse.ArgumentParser(description='Run MSSF_plus ablation suite.')
    parser.add_argument(
        '--group',
        type=str,
        default='main',
        choices=list(ABLATION_GROUPS.keys()) + ['all'],
        help='Ablation group to run',
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

    exps = get_experiments_for_group(args.group)
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
