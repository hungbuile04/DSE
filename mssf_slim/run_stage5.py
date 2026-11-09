"""
End-to-end runner for Stage 5 (`mssf_slim`):
  1. Pre-check: `verify_checksums.py --check` and `git diff main --stat` (enforcing isolation).
  2. Step 0 (Mandatory Parity Check): Run `cand_A` through `mssf_slim/train.py` on Fold 0,
     Split B (`seed=42`) from scratch (unless already verified in `step0_parity_check.json`
     and `--resume` is set). Assert `|MacroF1 - 0.4260| <= 0.005`; if violated, abort immediately.
  3. Run all 5 Stage 5 configurations (`cand_A`, `slim_nodec`, `slim_nodec_addw`, `slim_all`,
     `slim_all_cnnshared`) on Folds 0-2, Split B first, then Split A.
  4. Run `mssf_slim.summarize_results` and save verbatim output to
     `results_slim/stage5_slim_summarize_results.txt` (plus Google Drive backup if configured).
  5. Post-check: `verify_checksums.py --check` and `git diff main --stat`.
"""

import os
import sys
import json
import shutil
import argparse
import subprocess

from MSSF_plus.protocol import get_device, get_fold_list

from mssf_slim.configs import (
    DEFAULT_RESULTS_SLIM_DIR,
    SLIM_ORDER,
    get_slim_config,
)
from mssf_slim.train import (
    run_single_fold,
    run_experiment,
    _sync_method_dir,
)


REFERENCE_CAND_A_FOLD0_B_MACRO_F1 = 0.4260
STEP0_TOLERANCE = 0.005


def _run_isolation_checks(stage_label: str) -> None:
    print(f"\n{'=' * 95}")
    print(f"[{stage_label}] Running verify_checksums.py --check and git diff --stat ...")
    print(f"{'=' * 95}")
    subprocess.run(
        [sys.executable, "MSSF_plus/audit/verify_checksums.py", "--check"],
        check=True,
    )
    print("\n--- git diff --stat (working tree) ---")
    subprocess.run(["git", "diff", "--stat"], check=False)
    print("\n--- git diff main --stat (branch mssf-slim vs main) ---")
    subprocess.run(["git", "diff", "main", "--stat"], check=False)
    print(f"{'=' * 95}\n")


def _run_step0_parity_check(
    results_dir: str,
    drive_dir: str = None,
    device_str: str = None,
    resume: bool = True,
) -> dict:
    """
    Step 0 (Mandatory): Re-run `cand_A` through `mssf_slim/train.py` on Fold 0, Split B, `seed=42`.
    Verify `|valsel MacroF1 - 0.4260| <= 0.005`. Abort immediately if not satisfied.
    """
    os.makedirs(results_dir, exist_ok=True)
    s0_path = os.path.join(results_dir, 'step0_parity_check.json')
    if resume and drive_dir and not os.path.exists(s0_path):
        drive_s0 = os.path.join(drive_dir, 'step0_parity_check.json')
        if os.path.isfile(drive_s0):
            shutil.copy2(drive_s0, s0_path)

    method_dir = os.path.join(results_dir, 'cand_A_B')
    fold0_sum_path = os.path.join(method_dir, 'fold_0_summary.json')

    if resume and os.path.exists(s0_path) and os.path.exists(fold0_sum_path):
        with open(s0_path, 'r', encoding='utf-8') as f:
            s0_data = json.load(f)
        if s0_data.get('passed', False):
            print(
                f"=== [STEP 0 PARITY CHECK - RESUMED] cand_A Fold 0 Split B: "
                f"valsel MacroF1={s0_data['slim_cand_a_fold0_macro_f1']:.4f} "
                f"(ref={REFERENCE_CAND_A_FOLD0_B_MACRO_F1:.4f}, |diff|={s0_data['abs_diff']:.6f} <= {STEP0_TOLERANCE}) -> PASS ==="
            )
            return s0_data

    print(
        f"=== [STEP 0 MANDATORY PARITY CHECK] Running cand_A Fold 0, Split B (seed=42) via mssf_slim/train.py ==="
    )
    cfg = get_slim_config('cand_A')
    if device_str is not None:
        cfg['device'] = device_str
    device = get_device(cfg.get('device', None))

    summary = run_single_fold(
        fold=0,
        split_type='B',
        cfg=cfg,
        device=device,
        results_dir=results_dir,
        verbose=True,
    )
    _sync_method_dir(method_dir, drive_dir, 'cand_A_B')

    got_f1 = float(summary['valsel_metrics']['MacroF1'])
    abs_diff = abs(got_f1 - REFERENCE_CAND_A_FOLD0_B_MACRO_F1)
    passed = bool(abs_diff <= STEP0_TOLERANCE)

    s0_data = {
        'exp_name': 'cand_A',
        'split_type': 'B',
        'fold': 0,
        'seed': int(cfg.get('seed', 42)),
        'reference_macro_f1': REFERENCE_CAND_A_FOLD0_B_MACRO_F1,
        'slim_cand_a_fold0_macro_f1': got_f1,
        'abs_diff': abs_diff,
        'tolerance': STEP0_TOLERANCE,
        'passed': passed,
        'peak_gpu_mb': float(summary.get('peak_gpu_mb', 0.0)),
        'total_params': int(summary.get('total_params', 0)),
    }
    with open(s0_path, 'w', encoding='utf-8') as f:
        json.dump(s0_data, f, indent=2)
    if drive_dir:
        os.makedirs(drive_dir, exist_ok=True)
        shutil.copy2(s0_path, os.path.join(drive_dir, 'step0_parity_check.json'))

    print(
        f"=== [STEP 0 RESULT] cand_A Fold 0 Split B: "
        f"got={got_f1:.4f} vs ref={REFERENCE_CAND_A_FOLD0_B_MACRO_F1:.4f} | "
        f"|diff|={abs_diff:.6f} (threshold <= {STEP0_TOLERANCE:.4f}) -> {'PASS' if passed else 'FAIL'} ==="
    )
    if not passed:
        print(
            f"[CRITICAL ABORT] Step 0 parity check FAILED: |{got_f1:.4f} - {REFERENCE_CAND_A_FOLD0_B_MACRO_F1:.4f}| "
            f"= {abs_diff:.6f} > {STEP0_TOLERANCE:.4f}. Stopping immediately per protocol."
        )
        sys.exit(1)

    return s0_data


def main():
    parser = argparse.ArgumentParser(
        description='Run Stage 5 (mssf_slim): Step 0 parity check + 5 configs on Folds 0-2 (Split B then Split A).'
    )
    parser.add_argument('--splits', type=str, nargs='+', default=['B', 'A'], choices=['A', 'B'])
    parser.add_argument('--mode', type=str, default='dev', choices=['quick', 'dev', 'full'])
    parser.add_argument('--folds', type=int, nargs='*', default=None)
    parser.add_argument('--exps', type=str, nargs='*', default=None, help='Subset of SLIM_ORDER to run')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_SLIM_DIR)
    parser.add_argument('--drive_dir', type=str, default=None)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no_resume', action='store_false', dest='resume')
    parser.add_argument('--step0_only', action='store_true', help='Run only Step 0 parity check and exit')
    args = parser.parse_args()

    # 1. Pre-check
    _run_isolation_checks("PRE-CHECK")

    # 2. Step 0 Mandatory Parity Check
    _run_step0_parity_check(
        results_dir=args.results_dir,
        drive_dir=args.drive_dir,
        device_str=args.device,
        resume=args.resume,
    )
    if args.step0_only:
        _run_isolation_checks("POST-CHECK (STEP 0 ONLY)")
        return

    # 3. Run all requested configurations on Split B first, then Split A
    exps_to_run = args.exps if args.exps else list(SLIM_ORDER)
    overrides = {}
    if args.device is not None:
        overrides['device'] = args.device

    for split_type in args.splits:
        for exp_name in exps_to_run:
            run_experiment(
                exp_name=exp_name,
                split_type=split_type,
                mode=args.mode,
                custom_folds=args.folds,
                results_dir=args.results_dir,
                drive_dir=args.drive_dir,
                resume=args.resume,
                cli_overrides=overrides,
                verbose=True,
            )

    # 4. Run verbatim summarize_results.py and save to results_slim/
    folds_list = get_fold_list(mode=args.mode, custom_folds=args.folds)
    out_raw = os.path.join(args.results_dir, 'stage5_slim_summarize_results.txt')
    sum_cmd = [
        sys.executable, "-m", "mssf_slim.summarize_results",
        "--results_dir", args.results_dir,
        "--splits", *args.splits,
        "--folds", *[str(f) for f in folds_list],
        "--methods", *exps_to_run,
        "--out_raw", out_raw,
    ]
    if args.drive_dir:
        sum_cmd.extend(["--drive_dir", args.drive_dir])

    subprocess.run(sum_cmd, check=True)
    if args.drive_dir and os.path.exists(out_raw):
        os.makedirs(args.drive_dir, exist_ok=True)
        shutil.copy2(out_raw, os.path.join(args.drive_dir, 'stage5_slim_summarize_results.txt'))

    # 5. Post-check
    _run_isolation_checks("POST-CHECK")


if __name__ == '__main__':
    main()
