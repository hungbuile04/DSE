"""
One-shot Stage 7 Held-Out `test` Runner (`mssf_slim/run_stage7_test.py`).

Authorized post-review Stage 7 execution (`eval_test=True` strictly ONCE with all models,
hyperparameters, `val_select` early-stopping criteria, `val_select` baseline thresholds, and
`val_calib` C4 conformal quantiles frozen 100%):

  1. `--part slim`:
     Runs all 8 frozen `FINAL_10FOLD_EXPERIMENTS` (`slim_nodec`, `cand_A`, `slim_all_cnnshared`,
     `MSSF_v2_selfcorr`, and the 4 LOO configurations on `slim_nodec`) across 10 folds (`0..9`) on
     Split B & Split A with `eval_test=True`.
     Outputs are written to `results_slim/test_final/` (and synced to
     `/content/drive/MyDrive/DSE_results_slim_test`), keeping Stage 6 `val_select` files untouched.

  2. `--part baselines`:
     Dispatches the approved 10-fold external baselines with `eval_test=True` via read-only import
     of `MSSF_plus.baselines_rerun.run_all.dispatch_baseline`:
       - `MGPred_selfcorr`   (-> `MGPred_selfcorr` + `MGPred_selfcorr_thr`)
       - `SDPred_selfcorr`   (-> `SDPred_selfcorr` + `SDPred_selfcorr_thr`)
       - `DSGAT`             (-> `DSGAT` + `DSGAT_thr`)
       - `A3Net`             (-> `A3Net` + `A3Net_thr`)
       - `MVSL-DSF_selfcorr` (-> `MVSL-DSF_selfcorr` + `MVSL-DSF_selfcorr_thr`)
     Optional `--include_hstrans_3fold` runs `HSTrans` (+ `HSTrans_thr`) on Split B Folds `0, 1, 2`
     strictly last. All local baseline outputs stay inside `results_slim/test_final_baselines/`
     (synced to `/content/drive/MyDrive/DSE_results_v2_test`), preserving repo isolation.

  3. `--part summarize` (also run automatically at the end of `slim`, `baselines`, or `all`):
     Runs `mssf_slim.summarize_stage7_test` to export verbatim `stage7_test_summarize_results.txt`
     and `STAGE7_FINAL_TEST_REPORT.md` inside `results_slim/` (and mirrored to Google Drive).
"""

import os
import sys
import shutil
import argparse
import subprocess
from typing import List, Optional

from MSSF_plus.protocol import get_device, get_fold_list
from MSSF_plus.baselines_rerun.run_all import ALL_BASELINES, dispatch_baseline

from mssf_slim.configs import (
    DEFAULT_RESULTS_SLIM_DIR,
    FINAL_10FOLD_EXPERIMENTS,
    SLIM_EXPERIMENTS,
    get_slim_config,
)
from mssf_slim.run_stage5 import _run_isolation_checks
from mssf_slim.train import (
    _is_valid_json_file,
    _is_valid_npy_file,
    _sync_method_dir,
    _try_restore_completed_fold,
    run_experiment,
    run_single_fold,
)

APPROVED_STAGE7_10FOLD_BASELINES: List[str] = [
    'MGPred_selfcorr',
    'SDPred_selfcorr',
    'DSGAT',
    'A3Net',
    'MVSL-DSF_selfcorr',
]


def _scrub_corrupted_artifacts(root_dir: Optional[str]) -> None:
    """
    Scan `root_dir/*` and remove any 0-byte or truncated `.npy` / `.json` files left behind by
    an interrupted Colab session so resume never crashes on `np.load` or `json.load`.
    """
    if not root_dir or not os.path.isdir(root_dir):
        return
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if fname.endswith('.npy') and not _is_valid_npy_file(fpath):
                print(f"  [CLEANUP] Removing corrupted .npy file: {fpath}")
                os.remove(fpath)
            elif fname.endswith('.json') and not _is_valid_json_file(fpath):
                print(f"  [CLEANUP] Removing corrupted .json file: {fpath}")
                os.remove(fpath)


def run_stage7_slim_part(
    splits: List[str],
    mode: str,
    custom_folds: Optional[List[int]],
    exps_to_run: List[str],
    results_dir: str,
    drive_dir: Optional[str],
    device_arg: Optional[str],
    resume: bool,
) -> None:
    folds_list = get_fold_list(mode=mode, custom_folds=custom_folds)
    device = get_device(device_arg)
    overrides = {'eval_test': True}
    if device_arg is not None:
        overrides['device'] = device_arg

    _scrub_corrupted_artifacts(results_dir)
    _scrub_corrupted_artifacts(drive_dir)

    for split_type in splits:
        print(
            f"\n{'#' * 104}\n"
            f"# [STAGE 7 ONE-SHOT TEST] mssf_slim | SPLIT {split_type} | Folds={folds_list} | Configs={exps_to_run}\n"
            f"{'#' * 104}"
        )
        for fold in folds_list:
            for exp_name in exps_to_run:
                cfg = get_slim_config(exp_name, **overrides)
                assert cfg.get('eval_test', False) is True
                method_folder = f'{exp_name}_{split_type}'
                method_dir = os.path.join(results_dir, method_folder)

                if resume:
                    cached = _try_restore_completed_fold(
                        method_dir=method_dir,
                        drive_dir=drive_dir,
                        method_folder=method_folder,
                        fold=fold,
                        eval_test=True,
                    )
                    if cached is not None:
                        vm = cached['valsel_metrics']
                        tm = cached['test_metrics']
                        pgpu = float(cached.get('peak_gpu_mb', 0.0))
                        print(
                            f"  [RESUME] Skipping {method_folder} Fold {fold} "
                            f"(valsel F1={vm['MacroF1']:.4f} | TEST F1={tm['MacroF1']:.4f}, "
                            f"QWK={tm['QWK']:.4f}, PeakGPU={pgpu:.1f}MB)"
                        )
                        _sync_method_dir(method_dir, drive_dir, method_folder)
                        continue

                run_single_fold(
                    fold=fold,
                    split_type=split_type,
                    cfg=cfg,
                    device=device,
                    results_dir=results_dir,
                    verbose=True,
                )
                _sync_method_dir(method_dir, drive_dir, method_folder)

        for exp_name in exps_to_run:
            run_experiment(
                exp_name=exp_name,
                split_type=split_type,
                mode=mode,
                custom_folds=custom_folds,
                results_dir=results_dir,
                drive_dir=drive_dir,
                resume=True,
                cli_overrides=overrides,
                verbose=True,
            )


def run_stage7_baselines_part(
    splits: List[str],
    mode: str,
    custom_folds: Optional[List[int]],
    baselines: List[str],
    include_hstrans_3fold: bool,
    baseline_results_dir: str,
    baseline_drive_dir: Optional[str],
    device_arg: Optional[str],
    resume: bool,
) -> None:
    os.makedirs(baseline_results_dir, exist_ok=True)
    _scrub_corrupted_artifacts(baseline_results_dir)
    _scrub_corrupted_artifacts(baseline_drive_dir)

    for split_type in splits:
        for baseline in baselines:
            is_hstrans = (baseline == 'HSTrans')
            eff_mode = 'dev' if (is_hstrans and custom_folds is None and mode == 'full') else mode
            eff_folds = [0, 1, 2] if (is_hstrans and custom_folds is None) else custom_folds
            if is_hstrans and split_type != 'B' and custom_folds is None:
                print("  [INFO] Skipping HSTrans on Split A per approved 3-fold Split B budget.")
                continue

            print(
                f"\n{'=' * 100}\n"
                f"[STAGE 7 ONE-SHOT TEST] External Baseline: {baseline} | Split {split_type} | "
                f"Mode={eff_mode} | Folds={eff_folds if eff_folds is not None else 'mode-default'}\n"
                f"{'=' * 100}"
            )
            try:
                dispatch_baseline(
                    baseline=baseline,
                    split_type=split_type,
                    mode=eff_mode,
                    custom_folds=eff_folds,
                    results_dir=baseline_results_dir,
                    drive_dir=baseline_drive_dir,
                    resume=resume,
                    eval_test=True,
                    device=device_arg,
                )
            except ImportError as exc:
                print(
                    f"  [SKIP-ENV] Skipping {baseline} in this environment due to missing optional dependency: {exc}"
                )

    if include_hstrans_3fold and 'HSTrans' not in baselines and 'B' in splits:
        print(
            f"\n{'=' * 100}\n"
            f"[STAGE 7 ONE-SHOT TEST] External Baseline: HSTrans (Approved 3-Fold Split B `0..2`)\n"
            f"{'=' * 100}"
        )
        try:
            dispatch_baseline(
                baseline='HSTrans',
                split_type='B',
                mode='dev',
                custom_folds=[0, 1, 2],
                results_dir=baseline_results_dir,
                drive_dir=baseline_drive_dir,
                resume=resume,
                eval_test=True,
                device=device_arg,
            )
        except ImportError as exc:
            print(
                f"  [SKIP-ENV] Skipping HSTrans in this environment due to missing optional dependency: {exc}"
            )


def run_stage7_summarize_part(
    results_dir: str,
    drive_dir: Optional[str],
    baseline_results_dir: str,
    baseline_drive_dir: Optional[str],
    locked_valsel_dir: str,
    locked_valsel_drive_dir: Optional[str],
    splits: List[str],
    folds_list: List[int],
    exps_to_run: List[str],
) -> None:
    out_raw = os.path.join(DEFAULT_RESULTS_SLIM_DIR, 'stage7_test_summarize_results.txt')
    out_md = os.path.join(DEFAULT_RESULTS_SLIM_DIR, 'STAGE7_FINAL_TEST_REPORT.md')

    cmd = [
        sys.executable,
        "-m",
        "mssf_slim.summarize_stage7_test",
        "--results_dir",
        results_dir,
        "--baseline_results_dir",
        baseline_results_dir,
        "--locked_valsel_dir",
        locked_valsel_dir,
        "--splits",
        *splits,
        "--folds",
        *[str(f) for f in folds_list],
        "--methods",
        *exps_to_run,
        "--out_raw",
        out_raw,
        "--out_md",
        out_md,
    ]
    if drive_dir:
        cmd.extend(["--drive_dir", drive_dir])
    if baseline_drive_dir:
        cmd.extend(["--baseline_drive_dir", baseline_drive_dir])
    if locked_valsel_drive_dir:
        cmd.extend(["--locked_valsel_drive_dir", locked_valsel_drive_dir])

    subprocess.run(cmd, check=True)

    if drive_dir and os.path.isdir(drive_dir):
        for p in (out_raw, out_md):
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(drive_dir, os.path.basename(p)))


def main():
    parser = argparse.ArgumentParser(
        description='Stage 7 One-Shot Held-Out Test Runner (mssf_slim + external baselines + statistical report).'
    )
    parser.add_argument(
        '--part',
        type=str,
        default='slim',
        choices=['slim', 'baselines', 'summarize', 'all'],
        help='Which Stage 7 component to execute in this process.',
    )
    parser.add_argument('--splits', type=str, nargs='+', default=['B', 'A'], choices=['A', 'B'])
    parser.add_argument('--mode', type=str, default='full', choices=['quick', 'dev', 'full'])
    parser.add_argument('--folds', type=int, nargs='*', default=None)
    parser.add_argument('--exps', type=str, nargs='*', default=None)
    parser.add_argument(
        '--baselines',
        type=str,
        nargs='*',
        default=None,
        help='Subset of external baselines for --part baselines (defaults to APPROVED_STAGE7_10FOLD_BASELINES).',
    )
    parser.add_argument(
        '--include_hstrans_3fold',
        action='store_true',
        help='Also run 3-fold HSTrans (Folds 0..2 on Split B) after 10-fold baselines.',
    )
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument(
        '--results_dir',
        type=str,
        default=os.path.join(DEFAULT_RESULTS_SLIM_DIR, 'test_final'),
        help='Isolated directory inside results_slim/ for Stage 7 mssf_slim test outputs.',
    )
    parser.add_argument(
        '--drive_dir',
        type=str,
        default='/content/drive/MyDrive/DSE_results_slim_test',
    )
    parser.add_argument(
        '--baseline_results_dir',
        type=str,
        default=os.path.join(DEFAULT_RESULTS_SLIM_DIR, 'test_final_baselines'),
        help='Isolated directory inside results_slim/ for Stage 7 baseline test outputs.',
    )
    parser.add_argument(
        '--baseline_drive_dir',
        type=str,
        default='/content/drive/MyDrive/DSE_results_v2_test',
    )
    parser.add_argument(
        '--locked_valsel_dir',
        type=str,
        default=DEFAULT_RESULTS_SLIM_DIR,
    )
    parser.add_argument(
        '--locked_valsel_drive_dir',
        type=str,
        default='/content/drive/MyDrive/DSE_results_slim',
    )
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no_resume', action='store_false', dest='resume')
    args = parser.parse_args()

    exps_to_run = list(args.exps) if args.exps else list(FINAL_10FOLD_EXPERIMENTS)
    for exp_name in exps_to_run:
        if exp_name not in SLIM_EXPERIMENTS:
            raise KeyError(f"Unknown slim experiment {exp_name!r}. Valid: {list(SLIM_EXPERIMENTS.keys())}")

    baselines_to_run = list(args.baselines) if args.baselines else list(APPROVED_STAGE7_10FOLD_BASELINES)
    for b_name in baselines_to_run:
        if b_name not in ALL_BASELINES:
            raise KeyError(f"Unknown baseline {b_name!r}. Valid: {ALL_BASELINES}")

    # Enforce strict repository isolation check before running
    _run_isolation_checks(f"PRE-CHECK (STAGE 7 ONE-SHOT TEST: part={args.part})")

    folds_list = get_fold_list(mode=args.mode, custom_folds=args.folds)

    if args.part in ('slim', 'all'):
        run_stage7_slim_part(
            splits=args.splits,
            mode=args.mode,
            custom_folds=args.folds,
            exps_to_run=exps_to_run,
            results_dir=args.results_dir,
            drive_dir=args.drive_dir,
            device_arg=args.device,
            resume=args.resume,
        )

    if args.part in ('baselines', 'all'):
        run_stage7_baselines_part(
            splits=args.splits,
            mode=args.mode,
            custom_folds=args.folds,
            baselines=baselines_to_run,
            include_hstrans_3fold=args.include_hstrans_3fold,
            baseline_results_dir=args.baseline_results_dir,
            baseline_drive_dir=args.baseline_drive_dir,
            device_arg=args.device,
            resume=args.resume,
        )

    run_stage7_summarize_part(
        results_dir=args.results_dir,
        drive_dir=args.drive_dir,
        baseline_results_dir=args.baseline_results_dir,
        baseline_drive_dir=args.baseline_drive_dir,
        locked_valsel_dir=args.locked_valsel_dir,
        locked_valsel_drive_dir=args.locked_valsel_drive_dir,
        splits=args.splits,
        folds_list=folds_list,
        exps_to_run=exps_to_run,
    )

    _run_isolation_checks(f"POST-CHECK (STAGE 7 ONE-SHOT TEST: part={args.part})")


if __name__ == '__main__':
    main()
