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
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object
    '''
    #####--------------YOUR CODE STARTS HERE--------------#####
    train_set = spark.read.parquet(f"hdfs:/user/{userID}/train_set.parquet", header=True)
    train_set = train_set.drop("recording_mbid", "user_id")

    als = ALS(
        rank=100,  
        maxIter=15, 
        regParam=2, 
        userCol="user_id_index",  
        itemCol="mbid_index", 
        ratingCol="train_count",  
        coldStartStrategy="drop", 
        nonnegative=True, 
    )

    model = als.fit(train_set)

    user_recommendations = model.recommendForAllUsers(100)

    def extract_item_ids(recommendations):
        return [r.asDict()["mbid_index"] for r in recommendations]

   
    extract_item_ids_udf = udf(extract_item_ids, ArrayType(IntegerType()))

    user_recommendations_no_scores = user_recommendations.withColumn(
        "recommended_mbid_indices", extract_item_ids_udf(col("recommendations"))
    ).drop("recommendations")

    print('done')
    user_recommendations_no_scores.write.parquet(f"hdfs:/user/{userID}/recommendation_als.parquet")

    #user_recommendations_no_scores.show()
    

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommender').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function
    userID = os.environ['USER']
    main(spark, userID)
