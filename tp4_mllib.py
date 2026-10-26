"""
TP4 - Spark MLlib : clustering KMeans sur des données de ventes.

Pipeline : StringIndexer -> OneHotEncoder -> VectorAssembler -> KMeans (k=20)
Évaluation : coefficient de silhouette sur l'ensemble de test.

Usage :
    python tp4_mllib.py --data tp4_data
"""
import argparse

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format, lit, to_timestamp
from pyspark.sql.types import StringType


def main(data_dir: str, k: int, split_date: str) -> None:
    spark = SparkSession.builder.appName("TP4_MLlib").getOrCreate()

    # 1) Charger tous les CSV du dossier
    df = (spark.read.format("csv")
          .option("header", "true")
          .option("inferSchema", "true")
          .load(f"{data_dir}/*.csv"))

    # 2) Schéma
    df.printSchema()

    # 3) Valeurs manquantes -> 0
    df = df.na.fill(0)

    # 4) Colonne day_of_week
    if isinstance(df.schema["InvoiceDate"].dataType, StringType):
        df = df.withColumn("InvoiceDate", to_timestamp(col("InvoiceDate")))
    df = df.withColumn("day_of_week", date_format(col("InvoiceDate"), "EEEE"))

    # 5) Split train / test selon la date
    train_df = df.where(col("InvoiceDate") < lit(split_date))
    test_df = df.where(col("InvoiceDate") >= lit(split_date))
    print(f"Train: {train_df.count()} lignes | Test: {test_df.count()} lignes")

    # 6) StringIndexer (jour -> indice numérique)
    indexer = StringIndexer(inputCol="day_of_week",
                            outputCol="day_of_week_index",
                            handleInvalid="keep")

    # 7) OneHotEncoder (supprime l'ordre artificiel entre les jours)
    encoder = OneHotEncoder(inputCol="day_of_week_index",
                            outputCol="day_of_week_encoded")

    # 8) VectorAssembler
    assembler = VectorAssembler(
        inputCols=["UnitPrice", "Quantity", "day_of_week_encoded"],
        outputCol="features")

    # 9) Pipeline
    pipeline = Pipeline(stages=[indexer, encoder, assembler])

    # 10) fit : le StringIndexer apprend les valeurs uniques
    fitted_pipeline = pipeline.fit(train_df)

    # 11) Transformation du train
    transformed_train = fitted_pipeline.transform(train_df)

    # 12-13) KMeans
    kmeans = KMeans(k=k, seed=1, featuresCol="features", predictionCol="prediction")
    model = kmeans.fit(transformed_train)

    # 14) Prédictions sur le test
    predictions = model.transform(fitted_pipeline.transform(test_df))
    predictions.select("features", "prediction").show(10, truncate=False)

    # 15) Silhouette
    evaluator = ClusteringEvaluator(featuresCol="features",
                                    predictionCol="prediction",
                                    metricName="silhouette",
                                    distanceMeasure="squaredEuclidean")
    print("Coefficient de silhouette =", evaluator.evaluate(predictions))

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="tp4_data", help="Dossier contenant les CSV")
    parser.add_argument("--k", type=int, default=20, help="Nombre de clusters")
    parser.add_argument("--split-date", default="2010-12-13", help="Date de séparation train/test")
    args = parser.parse_args()
    main(args.data, args.k, args.split_date)
