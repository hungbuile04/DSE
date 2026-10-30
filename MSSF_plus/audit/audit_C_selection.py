"""Audit C — Epoch / checkpoint selection rule of the baselines that were run (MGPred, SDPred, MSSF).

Static evidence (line numbers refer to the files in the repo, before the notebook sed patches):
  MSSF   mssf.py L227-244: every epoch evaluates on the TEST loader, keeps metrics of the epoch with the
         highest test accuracy ("<Best Test>"). notebook1 cell 9 patch adds rating_te_best/pred_te_best
         inside that `if acc_te > acc_tested` block -> the SAVED preds are those of the best-test epoch.
  MGPred Ten_Fold_test.py L340-364: tests every epoch, dumps final_p{fold}.p when test RMSE improves,
         early-stops after `endure_count > 30` non-improving TEST evaluations. notebook1 cell 5 patches:
         test every 3 epochs, `endure_count > 2`, epochs=15  -> selection AND early stopping on test.
  SDPred up_ten_fold.py L165-190: early stopping uses TRAIN AUC/AUPR; test evaluated once after the
         loop (L190) -> no test-based selection. notebook1 cell 7 (Split A) comments L170-189 -> runs all
         epochs, last-epoch test. cell 12 (Split B) keeps L170-189 -> train-metric early stopping.

Numeric evidence (this script): parse the MSSF per-epoch test logs saved in notebook1 outputs and
quantify the optimism of 'best test epoch' vs. test-free alternatives (last epoch, mean of last 5).
"""
import os
import re
import json
import numpy as np
from common import REPO, save_json

NB = os.path.join(REPO, 'notebooks', 'notebook1_pytorch.ipynb')
PAT = re.compile(r'Epoch: (\d+) <Test> acc: ([\d.]+), weighted_f1: ([\d.]+), macro_f1: ([\d.]+), kappa: ([\d.]+)')
FOLD = re.compile(r'===== MSSF (?:Split B )?Fold (\d+)/10')


def cell_text(cell):
    out = []
    for o in cell.get('outputs', []):
        t = o.get('text') or o.get('data', {}).get('text/plain', '')
        out.append(''.join(t))
    return ''.join(out)


def parse(text):
    folds, cur = {}, None
    for line in text.splitlines():
        m = FOLD.search(line)
        if m:
            cur = int(m.group(1))
            folds[cur] = []
            continue
        m = PAT.search(line)
        if m and cur is not None:
            folds[cur].append([float(x) for x in m.groups()])
    return {k: np.array(v) for k, v in folds.items() if len(v)}


def summarise(folds, name):
    rows = []
    for k, a in sorted(folds.items()):
        ep, acc, wf1, mf1, kap = a.T
        b = int(np.argmax(acc))           # MSSF rule: first epoch with max test acc (strict >)
        rows.append({
            'fold': k, 'n_epochs': int(len(ep)), 'best_epoch_by_test_acc': int(ep[b]),
            'acc_best': acc[b], 'acc_last': acc[-1], 'acc_mean_last5': acc[-5:].mean(),
            'mf1_at_best_acc': mf1[b], 'mf1_last': mf1[-1], 'mf1_mean_last5': mf1[-5:].mean(),
            'mf1_oracle_max': mf1.max(),
            'kappa_at_best_acc': kap[b], 'kappa_last': kap[-1],
            'acc_epoch_to_epoch_std_last10': acc[-10:].std(),
        })
    print(f'\n=== MSSF {name}: {len(rows)} folds parsed ===')
    print('fold best_ep | acc best  last  last5 | mF1@best  last  last5  oracle | std(acc,last10)')
    for r in rows:
        print(f'{r["fold"]:4d} {r["best_epoch_by_test_acc"]:7d} | {r["acc_best"]:.4f} {r["acc_last"]:.4f} '
              f'{r["acc_mean_last5"]:.4f} | {r["mf1_at_best_acc"]:.4f} {r["mf1_last"]:.4f} '
              f'{r["mf1_mean_last5"]:.4f} {r["mf1_oracle_max"]:.4f} | {r["acc_epoch_to_epoch_std_last10"]:.4f}')
    agg = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key != 'fold'}
    print(f'MEAN        | {agg["acc_best"]:.4f} {agg["acc_last"]:.4f} {agg["acc_mean_last5"]:.4f} | '
          f'{agg["mf1_at_best_acc"]:.4f} {agg["mf1_last"]:.4f} {agg["mf1_mean_last5"]:.4f} '
          f'{agg["mf1_oracle_max"]:.4f}')
    print(f'Optimism of test-selection (acc): best - last = {agg["acc_best"] - agg["acc_last"]:+.4f}, '
          f'best - mean(last5) = {agg["acc_best"] - agg["acc_mean_last5"]:+.4f}')
    print(f'Macro-F1: @best-acc - last = {agg["mf1_at_best_acc"] - agg["mf1_last"]:+.4f}, '
          f'@best-acc - mean(last5) = {agg["mf1_at_best_acc"] - agg["mf1_mean_last5"]:+.4f}')
    return {'per_fold': rows, 'mean': agg}


def run():
    nb = json.load(open(NB))
    cells = nb['cells']
    src = lambda i: ''.join(cells[i]['source'])
    # sanity: which cells are MSSF A / B
    assert 'MSSF' in src(9) and 'SPLIT_TYPE' in src(9)
    assert 'MSSF Split B' in src(13)
    a = parse(cell_text(cells[9]))
    b = parse(cell_text(cells[13]))
    rep = {'MSSF_A': summarise(a, 'Split A'), 'MSSF_B': summarise(b, 'Split B')}

    # Static checks on notebook patches (string evidence)
    rep['static'] = {
        'MSSF_args_in_notebook': re.search(r'args_mssf = argparse.Namespace\((.*?)\)', src(9), re.S).group(1).split(),
        'MSSF_patch_saves_best_test_preds': 'rating_te_best = rating_te' in src(9) and 'pred_te_best = pred_te' in src(9),
        'MGPred_patch_test_every_3_epochs': 'epoch % 3 == 0' in src(5),
        'MGPred_patch_patience': "endure_count > 2" in src(5),
        'MGPred_epochs': re.search(r'epochs=(\d+)', src(5)).group(1),
        'SDPred_A_comments_L170_189': "170,189s/^/# /" in src(7),
        'SDPred_B_comments_L170_189': "170,189s/^/# /" in src(12),
        'SDPred_A_batch_size': re.search(r'batch_size=(\d+)', src(7)).group(1),
        'SDPred_B_batch_size': re.search(r'batch_size=(\d+)', src(12)).group(1),
    }
    print('\nStatic notebook checks:')
    for k, v in rep['static'].items():
        print(f'  {k}: {v}')
    save_json(rep, 'audit_C_selection.json')


if __name__ == '__main__':
    run()
