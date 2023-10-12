import sys
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, ntile, split, count, desc, round, sum, when, concat_ws,dense_rank, lit, udf
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.ml.recommendation import ALS

def main(spark, userID):
    #ground_truth_train = spark.read.parquet("hdfs:/user/{userID}/ground_truth_train_small.parquet",header=True)
    #ground_truth_val = spark.read.parquet(f"hdfs:/user/{userID}/ground_truth_val_small.parquet",header=True)
    ground_truth_test = spark.read.parquet(f"hdfs:/user/{userID}/ground_truth_test.parquet",header=True)
    recommendations_als = spark.read.parquet(f"hdfs:/user/{userID}/recommendation_als.parquet",header=True)

    #ground_truth_train_and_recommendations_als = ground_truth_train.join(recommendations_als, on="user_id_index")
    #ground_truth_val_and_recommendations_als = ground_truth_val.join(recommendations_als, on="user_id_index")
    ground_truth_test_and_recommendations_als = ground_truth_test.join(recommendations_als, on="user_id_index")
    #ground_truth_train_and_recommendations_als.write.parquet("hdfs:/user/aj3650_nyu_edu/ground_truth_train_and_recommendations_als.parquet")
    #ground_truth_val_and_recommendations_als.write.parquet(f"hdfs:/user/{userID}/ground_truth_val_and_recommendations_als_small.parquet")
    ground_truth_test_and_recommendations_als.write.parquet(f"hdfs:/user/{userID}/ground_truth_test_and_recommendations_als.parquet")

    #ground_truth_train_and_recommendations_als = spark.read.parquet(f"hdfs:/user/{userID}/ground_truth_train_and_recommendations_als.parquet", header=True)
    #rdd_train_als = ground_truth_train_and_recommendations_als.select('recommended_mbid_indices','ground_truth_train').rdd.map(lambda row: (row.recommended_mbid_indices, row.ground_truth_train))
    #metrics_train_als = RankingMetrics(rdd_train_als)
    #map_score_train_als = metrics_train_als.meanAveragePrecisionAt(10)
    #print('Mean Average Precision on Training',map_score_train_als)

    #ground_truth_val_and_recommendations_als = spark.read.parquet(f"hdfs:/user/{userID}/ground_truth_val_and_recommendations_als_small.parquet", header=True)
    ground_truth_test_and_recommendations_als = spark.read.parquet(f"hdfs:/user/{userID}/ground_truth_test_and_recommendations_als.parquet",header=True)
    rdd_test_als = ground_truth_test_and_recommendations_als.select('recommended_mbid_indices','ground_truth_test').rdd.map(lambda row: (row.recommended_mbid_indices, row.ground_truth_test))
    metrics_test_als = RankingMetrics(rdd_test_als)
    map_score_test_als = metrics_test_als.meanAveragePrecisionAt(100)
    print('Mean Average Precision on Test',map_score_test_als)

    

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommender').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function
    userID = os.environ['USER']
    main(spark, userID)
