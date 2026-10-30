"""
Leak-free baseline rerun wrappers under the unified 4-way evaluation protocol (`load_splits_v2`).

Baselines supported:
  - MGPred   (`run_mgpred.py`)
  - SDPred   (`run_sdpred.py`)
  - DSGAT    (`run_dsgat_a3net.py --model DSGAT`)
  - A3Net    (`run_dsgat_a3net.py --model A3Net`)
  - HSTrans  (`run_hstrans.py`)
  - MVSL-DSF (`run_mvsl.py`)
  - MSSF_v2  (delegates to `MSSF_plus.train.run_experiment('MSSF_v2', ...)`)
"""
