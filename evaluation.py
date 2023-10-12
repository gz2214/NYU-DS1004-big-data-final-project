import sys
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, ntile, split, count, desc, round, sum, when, concat_ws,dense_rank, lit
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.sql.types import DoubleType

def main(spark, userID):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object
    '''
    #####--------------YOUR CODE STARTS HERE--------------#####
    ground_truth_train_and_recommendations = spark.read.parquet(f"hdfs:/user/{userID}/ground_truth_train_and_recommendations_small.parquet", header=True)
    rdd_train = ground_truth_train_and_recommendations.select('recommendations','ground_truth_train').rdd.map(lambda row: (row.recommendations, row.ground_truth_train))
    metrics_train = RankingMetrics(rdd_train)
    map_score_train = metrics_train.meanAveragePrecisionAt(100)
    print('Mean Average Precision on Training',map_score_train)

    ground_truth_val_and_recommendations = spark.read.parquet(f"hdfs:/user/{userID}/ground_truth_val_and_recommendations_small.parquet", header=True)
    rdd_val = ground_truth_val_and_recommendations.select('recommendations','ground_truth_val').rdd.map(lambda row: (row.recommendations, row.ground_truth_val))
    metrics_val = RankingMetrics(rdd_val)
    map_score_val = metrics_val.meanAveragePrecisionAt(100)
    print('Mean Average Precision on Validation',map_score_val)


if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommender').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function
    userID = os.environ['USER']
    main(spark, userID)
