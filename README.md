# Spark MLlib : clustering KMeans sur des données de ventes

Projet réalisé dans le cadre du TP4 (Big Data / Spark). Il montre un pipeline complet
de Machine Learning avec **PySpark MLlib** : préparation des données, encodage de variables
catégorielles, clustering **KMeans** et évaluation par le **coefficient de silhouette**.

## Pipeline

```
CSV (plusieurs fichiers) -> DataFrame -> nettoyage -> day_of_week
   -> split train/test (date) -> StringIndexer -> OneHotEncoder
   -> VectorAssembler -> KMeans (k=20) -> prédictions -> silhouette
```

## Étapes principales

| Étape | Outil MLlib / Spark |
|---|---|
| Lecture de tous les CSV | `spark.read.csv("dossier/*.csv")` |
| Valeurs manquantes | `df.na.fill(0)` |
| Jour de la semaine | `date_format(col, "EEEE")` |
| Split temporel (avant / à partir de 2010-12-13) | `DataFrame.where` |
| Indexation du jour | `StringIndexer` |
| Suppression de l'ordre artificiel | `OneHotEncoder` |
| Vecteur de features (UnitPrice, Quantity, jour) | `VectorAssembler` |
| Enchaînement des étapes | `Pipeline` |
| Clustering | `KMeans(k=20)` |
| Évaluation | `ClusteringEvaluator` (silhouette) |

## Concepts clés

- **Estimator vs Transformer** : un Estimator (`StringIndexer`, `KMeans`) apprend avec `fit()`
  et produit un Transformer qui applique `transform()`.
- **Pourquoi OneHotEncoder ?** Le `StringIndexer` attribue des indices (ex. samedi = 6, lundi = 1),
  ce qui introduit un ordre artificiel. L'encodage one-hot représente chaque jour par un vecteur
  binaire, sans relation d'ordre.
- **Pourquoi `fit()` ?** Le `StringIndexer` doit connaître l'ensemble des valeurs uniques avant
  d'indexer. Le `fit` se fait uniquement sur le train, ce qui évite toute fuite d'information.
- **Silhouette** : entre -1 et 1. Proche de 1 : clusters bien séparés ; proche de 0 :
  clusters qui se chevauchent ; négatif : points probablement mal affectés.

## Lancer le projet

```bash
pip install -r requirements.txt
python tp4_mllib.py --data chemin/vers/tp4_data --k 20 --split-date 2010-12-13
```

Les données (un fichier CSV par jour, colonnes `InvoiceNo`, `StockCode`, `Description`,
`Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`) ne sont pas incluses dans ce dépôt.

## Pistes d'amélioration

- Standardiser `UnitPrice` et `Quantity` avec `StandardScaler` (échelles très différentes).
- Tester plusieurs valeurs de `k` et comparer les silhouettes (méthode du coude).
- Sauvegarder le pipeline et le modèle avec `.save()`.
