import sys
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,count
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.evaluation import RankingEvaluator
def main(spark, userID):
    recommendations_nn=spark.read.parquet(f'hdfs:/user/{userID}/recommendations_nn.parquet')
    ground_truth_train = spark.read.parquet(f"hdfs:/user/{userID}/ground_truth_train.parquet",header=True)
    ground_truth_val = spark.read.parquet(f"hdfs:/user/{userID}/ground_truth_val.parquet",header=True)
    ground_truth_test = spark.read.parquet(f"hdfs:/user/{userID}/ground_truth_test.parquet",header=True)
    
    ground_truth_train_and_recommendations_nn = ground_truth_train.join(recommendations_nn, on="user_id_index")
    ground_truth_val_and_recommendations_nn = ground_truth_val.join(recommendations_nn, on="user_id_index")
    ground_truth_test_and_recommendations_nn = ground_truth_test.join(recommendations_nn, on="user_id_index")

    rdd_train_nn = ground_truth_train_and_recommendations_nn.select('recommendations_nn','ground_truth_train').rdd.map(lambda row: (row.recommendations_nn, row.ground_truth_train))

    metrics_train_nn = RankingMetrics(rdd_train_nn)
    map_score_train_nn = metrics_train_nn.meanAveragePrecisionAt(100)
    print('Mean Average Precision on Training',map_score_train_nn)

    rdd_val_nn = ground_truth_val_and_recommendations_nn.select('recommendations_nn','ground_truth_val').rdd.map(lambda row: (row.recommendations_nn, row.ground_truth_val))
    metrics_val_nn = RankingMetrics(rdd_val_nn)
    map_score_val_nn = metrics_val_nn.meanAveragePrecisionAt(100)
    print('Mean Average Precision on Validation',map_score_val_nn)

    rdd_test_nn = ground_truth_test_and_recommendations_nn.select('recommendations_nn','ground_truth_test').rdd.map(lambda row: (row.recommendations_nn, row.ground_truth_test))
    metrics_test_nn = RankingMetrics(rdd_test_nn)
    map_score_test_nn = metrics_test_nn.meanAveragePrecisionAt(100)
    print('Mean Average Precision on Test',map_score_test_nn)

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('indexer').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function
    userID = os.environ['USER']
    main(spark, userID)
