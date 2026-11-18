# Legacy pipeline (archived)

This folder holds the first version of the project, kept unchanged for reference. It is not used
by the current pipeline in `src/`, and its figures are not quoted anywhere.

- scikit-learn Poisson and Gamma GLMs with a ridge penalty (`alpha = 0.01`) on seven banded factors.
- Isotonic calibration and seven hand-set loadings applied after the models.
- `dashboard.py`: the original Streamlit app, which fits the models when it starts.
- `build_excel_engine.py`: builds `artifacts/rating_engine.xlsx`, which reproduces the legacy price
  for 100 test policies. That workbook also holds two hand-built sheets, `Sample_Policy` and
  `Sensitivity_OneWay`, and stays in `artifacts/`.
- `plots/`: the charts the legacy pipeline produced.

`docs/AUDIT.md` explains why it was replaced. To run it (PowerShell, from the repo root):

```powershell
python -m pip install -r legacy\requirements.txt
python legacy\pricing_engine.py
streamlit run legacy\dashboard.py
```

`pricing_engine.py` writes its charts to `plots/` at the repo root, as it always did.
