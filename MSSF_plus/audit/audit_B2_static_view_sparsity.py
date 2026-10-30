"""Audit B2 — Intrinsic sparsity of the NON-profile drug views (all splits) and the
7 drugs that prepare_all_data.py could not map from 750 -> 757.

Motivation: Audit B showed many drugs have empty STITCH / pathway rows independently of the
split. This matters for C1 (availability indicator): 'missing view' is not only a
cold-start phenomenon of D9/D10.
"""
import os
import numpy as np
from common import static_views, drug_side_full, load_pkl, REPO, DRUG_VIEW_NAMES, save_json, N_DRUGS


def recompute_unmatched():
    """Same rule as shared_data/prepare_all_data.py find_drug_mapping() L55-73 (read-only)."""
    mg = np.array(load_pkl(os.path.join(REPO, 'MGPred', 'data', 'drug_side.pkl')))
    sd = np.array(load_pkl(os.path.join(REPO, 'SDPred', 'data', 'drug_side.pkl')))
    keys = {}
    groups = {}
    for j in range(sd.shape[0]):
        keys.setdefault(sd[j].tobytes(), j)           # FIRST equal row, as in prepare_all_data
        groups.setdefault(sd[j].tobytes(), []).append(j)
    mapping = [keys.get(mg[i].tobytes(), -1) for i in range(mg.shape[0])]
    dup_rows_757 = sd.shape[0] - len(keys)
    ambiguous = {i: groups[mg[i].tobytes()] for i in range(mg.shape[0])
                 if len(groups.get(mg[i].tobytes(), [])) > 1}
    print(f'750 drugs whose SE profile matches >1 row of 757 (feature rows copied from the first): {ambiguous}')
    print(f'  -> mapped 757 rows: {{i: mapping[i] for i in ambiguous}} = {({i: mapping[i] for i in ambiguous})}')
    return mapping, [i for i, m in enumerate(mapping) if m < 0], dup_rows_757, mg.shape, sd.shape


def run():
    dv, _ = static_views()
    full = drug_side_full()
    mapping, unmatched, dup757, mgs, sds = recompute_unmatched()
    print(f'MGPred drug_side {mgs}, SDPred drug_side {sds}; duplicate rows in 757: {dup757}')
    print(f'Unmatched 750->757 drugs: {unmatched}')
    rep = {'unmatched_drugs': unmatched, 'dup_rows_in_757': dup757, 'views': {}}
    print(f'\n{"view":22s} {"#drugs all-zero row":>20s} {"#drugs no off-diag":>19s}  unmatched-drug status')
    avail = np.zeros((N_DRUGS, len(DRUG_VIEW_NAMES) - 2), dtype=bool)
    static_names = [n for n in DRUG_VIEW_NAMES if 'profile' not in n]
    for j, n in enumerate(static_names):
        M = dv[n]
        off = M.copy()
        np.fill_diagonal(off, 0)
        allzero = np.abs(M).sum(1) == 0
        nooff = np.abs(off).sum(1) == 0
        avail[:, j] = ~nooff
        um = ['empty' if nooff[d] else 'ok' for d in unmatched]
        rep['views'][n] = {'n_all_zero': int(allzero.sum()), 'n_no_offdiag': int(nooff.sum()),
                           'unmatched_status': dict(zip(map(str, unmatched), um))}
        print(f'{n:22s} {allzero.sum():20d} {nooff.sum():19d}  {um}')
    n_avail = avail.sum(1)
    hist = {int(v): int((n_avail == v).sum()) for v in np.unique(n_avail)}
    print(f'\n#non-profile views with information per drug (out of {len(static_names)}): {hist}')
    rep['hist_n_available_static_views'] = hist
    # does missingness correlate with #known SEs? (informative-missingness check)
    nse = (full > 0).sum(1)
    stitch_missing = ~avail[:, static_names.index('D5_stitch_five')]
    print(f'mean #SE: STITCH-missing drugs={nse[stitch_missing].mean():.1f} '
          f'vs STITCH-present={nse[~stitch_missing].mean():.1f}')
    rep['mean_nSE_stitch_missing_vs_present'] = [float(nse[stitch_missing].mean()), float(nse[~stitch_missing].mean())]
    save_json(rep, 'audit_B2_static_view_sparsity.json')


if __name__ == '__main__':
    run()
