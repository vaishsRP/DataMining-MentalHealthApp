# Predicting Next-Day Mood from Smartphone Sensing

Data Mining Techniques, VU Amsterdam 2026. Group project by Priyanshi
Dhillon, Shithik Shaji, and Vaishanavi Mehta.

The question: can passive smartphone data (screen time, app usage,
activity, communication) predict tomorrow's mood in people with
depression, without asking them anything?

Short answer: only partially. Gradient boosting (weighted F1 0.537)
beat both a Random Forest baseline (0.495) and a tuned two-layer LSTM
(macro F1 0.49 vs 0.53). The strongest predictors were yesterday's mood
and affect, not app usage. And the "medium mood" class was nearly
unpredictable (F1 0.34) because it covers a band of 0.45 points on a
10-point scale. This is consistent with published work on similar data
(Asselbergs et al.). Worth keeping in mind when a mental-health app
promises mood detection from passive sensing alone.

Data: ecological momentary assessment from 27 participants over 31-69
days each. 1,268 daily records of passive sensor signals plus
self-reported mood, arousal, and valence.

## What we did

The analysis follows CRISP-DM end to end. The parts we spent the most
care on:

**Cleaning that respects what missingness means.** In this dataset a
missing app-usage value is not a data-collection failure: a zero-usage
day simply leaves no record. So `appCat.*` columns were zero-filled
rather than interpolated, while genuinely continuous signals got
time-series-appropriate imputation. Long gaps (over 4 days) were treated
as breaks rather than bridged, and one physically impossible value (a
negative app duration) was NaN-ed instead of dropping the whole row.

**Skew handling per user, not globally.** Screen time was winsorized at
each participant's own 95th percentile, and the heavy-tailed app-usage
columns were log1p-transformed, so one heavy user's tail doesn't define
"extreme" for everyone else.

**Leak-proof evaluation.** Splits are chronological within each
participant (final 20% as test), class thresholds for mood binning were
computed on the training set only, and features are lags, rolling
windows, and trends. No same-day peeks at the target.

**Two modelling approaches, compared honestly.** A non-temporal XGBoost
on windowed features versus a two-layer LSTM over 5-day sequences, both
grid-searched. The LSTM did not win. Short-term affect lags carry most
of the usable signal, and at 1,133 training instances a sequence model
has little room to show what it can do.

## Results

| Model | Weighted F1 | Macro F1 | Accuracy |
|---|---|---|---|
| Random Forest (top-10 features) | 0.495 | 0.485 | 0.502 |
| **XGBoost (all 24 features)** | **0.537** | **0.527** | **0.550** |
| LSTM (5-day windows) | — | 0.49 | — |

Per class (XGBoost): Low F1 0.62, Medium 0.34, High 0.62.
Misclassifications are almost all between adjacent classes, which makes
sense for an ordinal target. Feature importances are dominated by
lagged mood and circumplex (affect) variables; app-usage features are
secondary.

## Repo map

- `Task1A.ipynb` / `Task1B.ipynb` / `Task1C.ipynb` - EDA, cleaning,
  feature engineering
- `ML_Model_final.ipynb` - XGBoost / Random Forest classification
- `dl_model_lstm_FINAL.ipynb` - LSTM temporal classification
- `Report.pdf` - full write-up (16 pp.): methods, hyperparameter grids,
  association-rule mining extension, regression on the continuous
  target, and metric analysis
- `dataset_mood_smartphone.csv` - raw long-format data;
  `processed_data.csv`, `ML_Dataset.csv`, `DL_Dataset.csv` -
  intermediate artifacts

## Running it

```bash
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn
```

Open the notebooks in Jupyter and run in order: Task1A → Task1B →
Task1C → either model notebook. Intermediate CSVs are committed, so the
model notebooks also run standalone.
