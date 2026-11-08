# Report figures

Figures and tables used in the report.

- `lego/`, `poster/`: figures from the earlier ablations on the uncorrected data (A2 baseline and sampling runs).
- The final figures and tables (corrected data) are generated in `outputs/src_fixed/figures/`:
  `src/evaluate.py` writes the qualitative figures (`qualitative_<dataset>_<run>[_neighbour].png`), and
  `src/report_figures.py` writes the ablation tables (`table_ablation_<dataset>.tex`) and loss plots (`loss_<dataset>.pdf`).
  Copy the ones used in the report into this folder.