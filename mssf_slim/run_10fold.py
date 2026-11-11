"""
End-to-end runner for the approved 10-fold `val_select` evaluation (`eval_test=False` strictly!):
  - Runs 10 folds (`0..9`) on both Split B and Split A for the 8 approved configurations:
      1. `slim_nodec`              (Approved Final Model, 6.80M params, -36.51% vs cand_A)
      2. `cand_A`                  (Control with decoders, isolating exact 10-fold effect of removing decoders)
      3. `slim_all_cnnshared`      (Maximum-compression ablation variant, 2.24M params, -79.12%)
      4. `MSSF_v2_selfcorr`        (Base MSSF under Protocol v2 with self-correction)
      5. `slim_nodec_loo_no_vdrop` (LOO on `slim_nodec`: drop `vdrop`)
      6. `slim_nodec_loo_no_C2`    (LOO on `slim_nodec`: drop `C2`)
      7. `slim_nodec_loo_no_C5`    (LOO on `slim_nodec`: drop `C5`)
      8. `slim_nodec_loo_no_all`   (LOO on `slim_nodec`: drop `vdrop`, `C2`, and `C5`)

Key features:
  - Strictly enforces `eval_test=False` (test set remains untouched until post-review authorization).
  - Runs fold-major per split (`for fold in folds: for exp in exps`) so C2/C5 fold preprocessing is
    computed once per fold and shared in memory across configurations, while keeping all 8 configs
    lock-step synchronized if a Colab session is interrupted and resumed.
  - Automatically resumes completed folds (including Folds 0-2 of `slim_nodec`, `cand_A`, and
    `slim_all_cnnshared` from Stage 5) and cleans any 0-byte/corrupted `.npy` files.
  - Exports verbatim `valsel_10fold_summarize_results.txt` (and fold-matched baseline tables when
    `--baseline_drive_dir` or `--baseline_results_dir` is present).
"""

import os
import sys
import shutil
import argparse
import subprocess
from typing import List, Optional

from MSSF_plus.protocol import get_device, get_fold_list

from mssf_slim.configs import (
    DEFAULT_RESULTS_SLIM_DIR,
    FINAL_10FOLD_EXPERIMENTS,
    SLIM_EXPERIMENTS,
    get_slim_config,
)
from mssf_slim.run_stage5 import _run_isolation_checks
from mssf_slim.train import (
    _sync_method_dir,
    _try_restore_completed_fold,
    run_experiment,
    run_single_fold,
)


def main():
    parser = argparse.ArgumentParser(
        description='Run 10-fold val_select evaluation (eval_test=False) for the 8 final & LOO configurations.'
    )
    parser.add_argument('--splits', type=str, nargs='+', default=['B', 'A'], choices=['A', 'B'])
    parser.add_argument('--mode', type=str, default='full', choices=['quick', 'dev', 'full'])
    parser.add_argument('--folds', type=int, nargs='*', default=None)
    parser.add_argument(
        '--exps',
        type=str,
        nargs='*',
        default=None,
        help='Optional subset of configurations to run (defaults to all 8 FINAL_10FOLD_EXPERIMENTS).',
    )
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_SLIM_DIR)
    parser.add_argument('--drive_dir', type=str, default=None)
    parser.add_argument(
        '--baseline_results_dir',
        type=str,
        default=None,
        help='Optional local directory containing Stage 6 external baseline .npy files (e.g., results_v2).',
    )
    parser.add_argument(
        '--baseline_drive_dir',
        type=str,
        default='/content/drive/MyDrive/DSE_results_v2',
        help='Optional Drive directory containing Stage 6 external baseline .npy files (auto-ignored if absent).',
    )
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no_resume', action='store_false', dest='resume')
    args = parser.parse_args()

    exps_to_run: List[str] = list(args.exps) if args.exps else list(FINAL_10FOLD_EXPERIMENTS)
    for exp_name in exps_to_run:
        if exp_name not in SLIM_EXPERIMENTS:
            raise KeyError(
                f"Unknown experiment {exp_name!r}. Valid options: {list(SLIM_EXPERIMENTS.keys())}"
            )

    # 1. Pre-check isolation & checksums
    _run_isolation_checks("PRE-CHECK (10-FOLD VAL_SELECT)")

    folds_list = get_fold_list(mode=args.mode, custom_folds=args.folds)
    device = get_device(args.device)
    overrides = {'eval_test': False}
    if args.device is not None:
        overrides['device'] = args.device

    # 2. Fold-major execution per split (maximizes _FOLD_PREP_CACHE hits & keeps all configs fold-synchronized)
    for split_type in args.splits:
        print(
            f"\n{'#' * 100}\n"
            f"# SPLIT {split_type} | Folds={folds_list} | Configs={exps_to_run} | eval_test=False (STRICT LOCK)\n"
            f"{'#' * 100}"
        )
        for fold in folds_list:
            for exp_name in exps_to_run:
                cfg = get_slim_config(exp_name, **overrides)
                assert cfg.get('eval_test', False) is False, "CRITICAL: eval_test MUST remain False!"
                method_folder = f'{exp_name}_{split_type}'
                method_dir = os.path.join(args.results_dir, method_folder)

                if args.resume:
                    cached = _try_restore_completed_fold(
                        method_dir=method_dir,
                        drive_dir=args.drive_dir,
                        method_folder=method_folder,
                        fold=fold,
                        eval_test=False,
                    )
                    if cached is not None:
                        vm = cached['valsel_metrics']
                        pgpu = float(cached.get('peak_gpu_mb', 0.0))
                        print(
                            f"  [RESUME] Skipping {method_folder} Fold {fold} "
                            f"(valsel MacroF1={vm['MacroF1']:.4f}, QWK={vm['QWK']:.4f}, PeakGPU={pgpu:.1f}MB)"
                        )
                        _sync_method_dir(method_dir, args.drive_dir, method_folder)
                        continue

                run_single_fold(
                    fold=fold,
                    split_type=split_type,
                    cfg=cfg,
                    device=device,
                    results_dir=args.results_dir,
                    verbose=True,
                )
                _sync_method_dir(method_dir, args.drive_dir, method_folder)

        # Finalize experiment-level summary_{mode}.json & config.json via instant resume pass
        for exp_name in exps_to_run:
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

    # 3. Export verbatim summary reports (10-fold val_select + optional Fold 0-2 matched baseline table)
    b_drive: Optional[str] = (
        args.baseline_drive_dir
        if (args.baseline_drive_dir and os.path.isdir(args.baseline_drive_dir))
        else None
    )
    b_local: Optional[str] = (
        args.baseline_results_dir
        if (args.baseline_results_dir and os.path.isdir(args.baseline_results_dir))
        else None
    )

    out_raw_main = os.path.join(args.results_dir, 'valsel_10fold_summarize_results.txt')
    sum_cmd = [
        sys.executable, "-m", "mssf_slim.summarize_results",
        "--results_dir", args.results_dir,
        "--splits", *args.splits,
        "--folds", *[str(f) for f in folds_list],
        "--methods", *exps_to_run,
        "--out_raw", out_raw_main,
    ]
    if args.drive_dir:
        sum_cmd.extend(["--drive_dir", args.drive_dir])
    if b_local:
        sum_cmd.extend(["--baseline_results_dir", b_local])
    if b_drive:
        sum_cmd.extend(["--baseline_drive_dir", b_drive])

    subprocess.run(sum_cmd, check=True)
    if args.drive_dir and os.path.exists(out_raw_main):
        os.makedirs(args.drive_dir, exist_ok=True)
        shutil.copy2(out_raw_main, os.path.join(args.drive_dir, 'valsel_10fold_summarize_results.txt'))

    if (b_local or b_drive) and set([0, 1, 2]).issubset(set(folds_list)):
        out_raw_f02 = os.path.join(args.results_dir, 'valsel_fold0_2_matched_baselines.txt')
        f02_cmd = [
            sys.executable, "-m", "mssf_slim.summarize_results",
            "--results_dir", args.results_dir,
            "--splits", *args.splits,
            "--folds", "0", "1", "2",
            "--methods", *exps_to_run,
            "--no_conformal",
            "--out_raw", out_raw_f02,
        ]
        if args.drive_dir:
            f02_cmd.extend(["--drive_dir", args.drive_dir])
        if b_local:
            f02_cmd.extend(["--baseline_results_dir", b_local])
        if b_drive:
            f02_cmd.extend(["--baseline_drive_dir", b_drive])
        subprocess.run(f02_cmd, check=True)
        if args.drive_dir and os.path.exists(out_raw_f02):
            shutil.copy2(out_raw_f02, os.path.join(args.drive_dir, 'valsel_fold0_2_matched_baselines.txt'))

    # 4. Post-check isolation & checksums
    _run_isolation_checks("POST-CHECK (10-FOLD VAL_SELECT)")


if __name__ == '__main__':
    main()
