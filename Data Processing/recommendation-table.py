import sys
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, ntile, split, count, desc, round, sum, when, concat_ws,dense_rank, lit
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def main(spark, userID):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object
    '''
    #####--------------YOUR CODE STARTS HERE--------------#####
    #Train Set Ground Truth
    '''
    train_set = spark.read.parquet(f"hdfs:/user/{userID}/train_set.parquet", header=True) 

    window_spec = Window.partitionBy("user_id_index").orderBy(F.desc("train_count"), F.asc("mbid_index"))#track_artist_key
    df_with_rank = train_set.withColumn("rank", F.row_number().over(window_spec))
    df_top_100 = df_with_rank.filter(F.col("rank") <= 100)
    df_user_top_songs = df_top_100.groupBy("user_id_index").agg(F.collect_list("mbid_index").alias("ground_truth_train"))
    df_user_top_songs.write.parquet(f"hdfs:/user/{userID}/ground_truth_train.parquet")
    df_user_top_songs.show()

    #Val Set Ground Truth
    validation_set = spark.read.parquet(f"hdfs:/user/{userID}/validation_set.parquet", header=True) 

    unique_user_ids = validation_set.select("user_id_index").distinct()

    window_spec = Window.partitionBy("user_id_index").orderBy(F.desc("val_count"), F.asc("mbid_index"))#track_artist_key
    df_with_rank = validation_set.withColumn("rank", F.row_number().over(window_spec))
    df_top_100 = df_with_rank.filter(F.col("rank") <= 100)
    df_user_top_songs = df_top_100.groupBy("user_id_index").agg(F.collect_list("mbid_index").alias("ground_truth_val"))
    df_user_top_songs.write.parquet(f"hdfs:/user/{userID}/ground_truth_val.parquet")
    df_user_top_songs.show()
    '''

    
    # test set ground truth
    test_set=spark.read.parquet(f'hdfs:/user/{userID}/test_set.parquet',header=True)
    unique_user_ids = test_set.select("user_id_index").distinct()

    window_spec = Window.partitionBy("user_id_index").orderBy(F.desc("play_count"), F.asc("mbid_index"))
    df_with_rank = test_set.withColumn("rank", F.row_number().over(window_spec)).withColumnRenamed('play_count', 'test_count')
    df_top_100 = df_with_rank.filter(F.col("rank") <= 100)
    df_user_top_songs = df_top_100.groupBy("user_id_index").agg(F.collect_list("mbid_index").alias("ground_truth_test"))
    df_user_top_songs.write.parquet(f"hdfs:/user/{userID}/ground_truth_test.parquet")
    df_user_top_songs.show()
    print('recommend')
    recommendations = spark.read.parquet(f"hdfs:/user/{userID}/recommendations_train.parquet",header=True)
    ground_truth_test = spark.read.parquet(f"hdfs:/user/{userID}/ground_truth_test.parquet",header=True)
    recommendations = spark.read.parquet(f"hdfs:/user/{userID}/recommendations_train.parquet",header=True)
    ground_truth_test_and_recommendations = ground_truth_test.join(recommendations, on="user_id_index")
    ground_truth_test_and_recommendations.write.parquet(f"hdfs:/user/{userID}/ground_truth_test_and_recommendations.parquet")
    ground_truth_test_and_recommendations.show()
    '''
    #Recommendations
    top_songs = spark.read.parquet(f"hdfs:/user/{userID}/popular_train.parquet",header=True)
    top_100_songs = [lit(song) for song in top_songs.select("mbid_index").rdd.flatMap(lambda x: x).collect()]
    unique_user_ids_with_top_100 = unique_user_ids.withColumn("recommendations", F.array(top_100_songs))
    unique_user_ids_with_top_100.write.parquet(f"hdfs:/user/{userID}/recommendations_train.parquet")
    unique_user_ids_with_top_100.show()
    
    
    #Merge ground truth and recommendations into one dataframe
    ground_truth_train = spark.read.parquet(f"hdfs:/user/{userID}/ground_truth_train.parquet",header=True)
    ground_truth_val = spark.read.parquet(f"hdfs:/user/{userID}/ground_truth_val.parquet",header=True)
    recommendations = spark.read.parquet(f"hdfs:/user/{userID}/recommendations_train.parquet",header=True)
    ground_truth_val_and_recommendations = ground_truth_val.join(recommendations, on="user_id_index")
    ground_truth_train_and_recommendations = ground_truth_train.join(recommendations, on="user_id_index")
    ground_truth_val_and_recommendations.write.parquet(f"hdfs:/user/{userID}/ground_truth_val_and_recommendations.parquet")
    ground_truth_train_and_recommendations.write.parquet(f"hdfs:/user/{userID}/ground_truth_train_and_recommendations.parquet")
    ground_truth_val_and_recommendations.show()
    ground_truth_train_and_recommendations.show()
    '''
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommender').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function
    userID = os.environ['USER']
    main(spark, userID)
