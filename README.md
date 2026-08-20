# Humanitar Data Science.py

Data science project developed for Humanitar to analyze healthcare appointment demand and build forecasting models by service, department, and hour of the day.

> The original dataset is confidential and is not included in this repository.

## Project objective

The objective is to support operational planning by estimating future appointment demand across healthcare services. The analysis explores how demand varies over time, across departments, and between healthcare facilities.

## Project scope

- Integrated appointment records with hospital structure data.
- Analyzed appointment demand by month, hour, day of the week, department, and hospital.
- Examined the effect of holidays on appointment demand.
- Built forecasting workflows for multiple future time horizons.
- Compared model predictions with a historical forecasting baseline.

## Models

The project evaluates the following models:

- Linear Regression
- XGBoost Regressor

The models are trained using historical appointment data and evaluated through time-based backtesting, which keeps the chronological order of the data.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- XGBoost

## Repository structure

```text
.
├── Humanitar Data Science.py
├── README.md
└── .gitignore
```

## Author

Lorenzo de Almeida Torriero
