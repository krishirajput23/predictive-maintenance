# Predictive Maintenance

A model that looks at daily sensor readings from machines and tells you which ones are likely to fail soon, so maintenance can check them before they break down. There's a small Streamlit app on top to try it out.

## The data

Daily readings for 1,169 machines, Jan to Nov 2015 (about 124k rows). Each row has nine anonymous sensor metrics, and a `failure` flag. Only 106 rows are failures (0.085%), one per machine at most.

Things to keep in mind: the metrics have no names or units, so we can't say what they physically mean. `metric8` is just a copy of `metric7`, so it's dropped. With so few failures, every number below is noisy.

## What I did

- **Features:** the raw metrics, a few date fields, how long a machine has been in the data, and for each metric a z-score against that machine's own past readings (past only, so nothing leaks from the future).
- **Split:** by machine, not by row, so the test set is made of machines the model has never seen.
- **Models:** logistic regression, decision tree, random forest, gradient boosting and XGBoost.
- **Metric:** PR-AUC. Accuracy is useless here since "never fails" already scores 99.9%.
- **Threshold:** picked on the validation set, not left at 0.5.

## Results

XGBoost won. On the test machines:

| | |
| --- | --- |
| PR-AUC | 0.134 (random guessing gives 0.0009) |
| ROC-AUC | 0.927 |
| Precision / recall | 0.20 / 0.31 |
| Caught / missed / false alarms | 5 / 11 / 20 |

So it ranks risky machines far better than chance, but at the chosen threshold it still misses most failures. Lowering the threshold catches more at the cost of more false alarms; the right setting depends on what a surprise breakdown costs compared to an unnecessary inspection. The app has a slider for this.

Full comparison of all five models is in `outputs/model_comparison.csv`.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app has three tabs: score a single machine (from the dataset or your own CSV), a risk ranking of the held-out test machines, and the model results with plots.

Other commands:

```bash
python scripts/train.py      # retrain everything (a few minutes)
python scripts/predict.py    # quick command line demo
python -m unittest discover -s tests -v
```

For your own CSV, use the columns `date` (m/d/YYYY), `device`, `metric1` to `metric9`, and include each machine's recent history, because the deviation features are built from the rows you supply.

## Folder layout

```
app.py          Streamlit app
src/            data prep, models, evaluation
scripts/        train.py, predict.py
notebooks/      walkthrough notebook
tests/          smoke tests
outputs/        saved model, metrics, plots
data/           the dataset
```

## Limits

- Only 16 failures in the test set, so treat the scores as rough.
- Trained on one fleet with anonymous metrics. Retrain on new data before using it anywhere real, and watch for drift.
- It's decision support. It should schedule inspections, not shut machines down on its own.
