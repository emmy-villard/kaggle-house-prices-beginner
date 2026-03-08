# Kaggle House Prices - Portfolio Project

Projet ML de regression sur la competition Kaggle
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

## Resultat

- Score Kaggle: `0.13234`
- Classement: `1614 / 4102`

Sur cette competition, le score est une erreur calculee sur `log(SalePrice)` (type RMSLE):
plus le score est bas, meilleures sont les predictions.

## Ce que ce projet demontre

- Construction d'un pipeline de preprocessing reproductible (`scikit-learn Pipeline` + `ColumnTransformer`)
- Gestion des valeurs manquantes et encodage One-Hot des variables categorielles
- Benchmark de plusieurs modeles de regression
- Selection d'un modele final et generation d'une soumission Kaggle

## Stack

- `Python`, `pandas`, `numpy`
- `scikit-learn`
- `xgboost`, `catboost`
- `tensorflow` / `keras` / `keras-tuner` (experimentation)

## Pipeline (end-to-end)

1. `preprocessing/build_preprocessor.py`
: entraine et sauvegarde `preprocess.joblib`
2. `preprocessing/apply_preprocess.py`
: genere `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, `to_predict.csv`
3. `modeling/modeling.py`
: entraine un `XGBRegressor` et sauvegarde `modeling/model.joblib`
4. `modeling/data_prediction.py`
: cree `submission.csv` (`Id`, `SalePrice`)

## Modeles compares (notebook)

Dans `modeling/modeling_exploration.ipynb`:

- CatBoost: `R2 = 0.88`
- RidgeCV: `R2 = 0.63`
- XGBoost: `R2 = 0.89` (modele retenu)
- Reseau de neurones: `R2 = -1.4567`

## Données

- `train.csv`: 1460 lignes, 81 colonnes
- `test.csv`: 1459 lignes, 80 colonnes
- apres preprocessing: 289 features

## Lancer le projet

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python preprocessing/build_preprocessor.py
python preprocessing/apply_preprocess.py
python modeling/modeling.py
python modeling/data_prediction.py
```

## Fichiers cles

- `preprocessing/preprocessing_exploration.ipynb`
- `preprocessing/build_preprocessor.py`
- `preprocessing/apply_preprocess.py`
- `modeling/modeling_exploration.ipynb`
- `modeling/modeling.py`
- `modeling/data_prediction.py`