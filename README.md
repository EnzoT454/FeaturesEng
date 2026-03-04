# Interprétabilité de Modèle et Ingénierie des Caractéristiques

**IFT-3700 — Science des Données**

## Aperçu du projet

Ce projet explore plusieurs concepts essentiels en science des données :

* **Interprétabilité des modèles**
* **Explicabilité des prédictions**
* **Détection et suppression des valeurs aberrantes**
* **Sélection et ingénierie des caractéristiques**

Deux ensembles de données sont utilisés :

1. **Hospital Readmission Dataset** → analyser et expliquer les prédictions d’un modèle de classification.
2. **New York Taxi Dataset** → nettoyer les données, détecter les valeurs aberrantes et créer de nouvelles caractéristiques.

Le projet utilise principalement :

* **Random Forest**
* **Permutation Feature Importance**
* **Partial Dependence Plots**
* **SHAP explanations**
* **Méthodes de nettoyage de données**

---

## Structure du projet

```text
.
├── explainability_pipeline.py        # Implémentation des fonctions ML
├── explainability_analysis.ipynb     # Notebook d'analyse et visualisations
├── data
│   ├── hospital.csv
│   └── ny_taxi.csv
├── requirements.txt
└── README.md
```
---
## Installation

Créer un environnement virtuel et installer les dépendances.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Exécution du projet

```bash
jupyter lab
```

Puis ouvrir : explainability_analysis.ipynb

Le notebook exécute toutes les étapes :

* Chargement des données
* Prétraitement
* Entraînement du modèle
* Évaluation
* Interprétation
* Visualisation

# Rapport d'Analyse : Interprétabilité et Ingénierie des Caractéristiques

## Partie 1 — Interprétabilité du modèle (Hospital Dataset)

**Dataset utilisé :** `hospital.csv`  
**Objectif :** Prédire si un patient sera réadmis à l’hôpital.

### Prétraitement des données
La variable cible `is_readmitted` est encodée en valeurs numériques à l’aide de **LabelEncoder**.
* **Fonction utilisée :** `encode_target_column()`
* **Transformation :** `True` → 1, `False` → 0

### Séparation des données
Les données sont divisées en ensemble d'entraînement et ensemble de validation via `train_test_split()` :
* `test_size = 0.2`
* `random_state = 42`

### Entraînement du modèle
Un modèle **RandomForestClassifier** est entraîné via la fonction `train_random_forest()`. Les forêts aléatoires sont robustes et performantes pour les problèmes de classification.

### Évaluation du modèle
Les performances sont mesurées via `evaluate_model()` en utilisant :
* **L'Accuracy**
* **Le Classification Report** (précision, rappel, F1-score)

### Importance des caractéristiques (Permutation Importance)
La **Permutation Importance** mesure l’importance d’une variable en observant la baisse de performance du modèle lorsqu’on mélange ses valeurs.
* **Implémentation :** `eli5.sklearn.PermutationImportance`
* Cette méthode est **indépendante du modèle**.

### Graphiques de dépendance partielle
Les **Partial Dependence Plots (PDP)** montrent comment une variable influence la prédiction du modèle.
* **Bibliothèque :** `sklearn.inspection.PartialDependenceDisplay`

### Taux de réadmission vs durée d'hospitalisation
Vérification de la cohérence de la variable `time_in_hospital` via la fonction `plot_mean_readmission_vs_time()`.


---

## Partie 2 — Détection des valeurs aberrantes (NY Taxi Dataset)

**Dataset utilisé :** `ny_taxi.csv`  
**Objectif :** Améliorer la qualité des données avant l'entraînement d'un modèle.

### Suppression des valeurs aberrantes (méthode IQR)
Les valeurs aberrantes sont détectées avec la méthode **IQR (Interquartile Range)** :

$$IQR = Q3 - Q1$$

**Limites de détection :**
$$[ Q1 - 1.5 \times IQR , Q3 + 1.5 \times IQR ]$$

* **Fonction :** `remove_outliers_iqr()`



### Sélection des caractéristiques
La **Permutation Importance** identifie les variables clés comme les coordonnées géographiques, la distance du trajet et le nombre de passagers.

### Ingénierie des caractéristiques
Création de deux nouvelles variables via `add_absolute_coordinate_changes()` :
* `abs_lon_change` = $|dropoff\_longitude - pickup\_longitude|$
* `abs_lat_change` = $|dropoff\_latitude - pickup\_latitude|$

Ces variables représentent approximativement la distance parcourue par le taxi.

---

## Technologies utilisées
* **Langage :** Python (Pandas, NumPy, Scikit-learn)
* **Interprétabilité :** ELI5, SHAP
* **Visualisation :** Matplotlib, Seaborn
* **Environnement :** Jupyter Notebook

---

## Conclusion
Ce projet met en évidence l’importance de comprendre et d’expliquer les modèles de machine learning :
1. **Identifier** les variables importantes.
2. **Visualiser** l’effet des caractéristiques sur les prédictions.
3. **Expliquer** des prédictions individuelles grâce à SHAP.
4. **Nettoyer** les données avec la méthode IQR.
5. **Créer** des caractéristiques pertinentes (Feature Engineering) pour améliorer la performance.
