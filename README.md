# Titanic Dataset — Exploratory Analysis with pandas

Academic project by **Ludovic Delot Bravo**, undergraduate student in the
**Licenciatura en Inteligencia de Negocios (Business Intelligence)** at
**Tecnológico de Monterrey (Tec de Monterrey)**.

- Course: *Programación para Negocios* (Programming for Business), Semester 1 (S1)
- Assignment: *Actividad 7 — Análisis del dataset Titanic*
- Tools: Python, pandas, matplotlib (originally developed in Google Colaboratory)

## What this project does

`titanic_analysis.py` performs an exploratory data analysis (EDA) of the
classic Titanic passenger dataset using pandas:

1. **Load & inspect** the dataset (`info()`, `shape`, `head()`, `tail()`).
2. **Clean missing values**: fills `Age` with the column mean, `Cabin` with
   a placeholder ("Sin Registro"), and `Embarked` with the most common port
   ("S").
3. **Recode columns**: maps `Pclass` (1/2/3) to readable labels ("Clase
   Alta", "Clase Media", "Clase Baja") and renames `Sex` to `Gender`.
4. **Descriptive statistics**: survivor counts, passengers per class, average
   fare and age by group, cross-tabulations of survival by gender/class,
   and passenger titles.
5. **Feature engineering**: buckets `Age` into ranges (Jóvenes, Adultos,
   Adultos Mayores) and creates a `Family_Size` feature from `SibSp` +
   `Parch`.
6. **Visualizations**: histogram of survival, horizontal bar chart of
   survival by gender, line chart of survival by class, and a pie chart of
   passengers by class.

Console output is in Spanish, matching the original coursework submission.

## How to run it

```bash
pip install -r requirements.txt
python titanic_analysis.py
```

The script reads `data/train.csv` and prints its analysis to the console,
saving chart images (`.png`) and an updated CSV (`Titanic_actualizado.csv`)
in the working directory. These generated outputs are excluded from version
control (see `.gitignore`) — re-run the script to reproduce them.

## Key findings (from the original submission)

- Survival was strongly associated with **gender**: women had a
  substantially higher survival rate than men.
- Survival was strongly associated with **passenger class**: first-class
  ("Clase Alta") passengers survived at a higher rate than third-class
  ("Clase Baja") passengers.
- Average fare paid increased with passenger class, as expected.
- Most passengers were traveling without large families aboard
  (`Family_Size` was low for the majority).

## Dataset attribution

The data (`data/train.csv`) is the well-known **Titanic - Machine Learning
from Disaster** dataset, originally published by
[Kaggle](https://www.kaggle.com/competitions/titanic/data) and widely used
as an open, public teaching dataset. It contains historical, publicly
available passenger records from the RMS Titanic (1912) — no private or
sensitive personal data was added. The version used here includes the same
891 passenger records as the standard Kaggle `train.csv`, with `Title` and
`Last Name` split out as additional columns for the exercise.

The dataset is provided here for educational/reproducibility purposes
alongside the coursework it was used in.

## Repository structure

```
.
├── README.md
├── titanic_analysis.py     # main analysis script
├── data/
│   └── train.csv            # Titanic dataset (Kaggle, public)
├── requirements.txt
└── .gitignore
```
