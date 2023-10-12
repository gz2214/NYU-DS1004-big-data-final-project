import sys
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, ntile, split, count, desc, round, sum, when, concat_ws,dense_rank

def main(spark, userID):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object
    '''
    #####--------------YOUR CODE STARTS HERE--------------#####
    '''   
    interactionsTrainSmall = spark.read.parquet(f'hdfs:/user/{userID}/interactionsTrain.parquet',header=True)
    user_song_counts = interactionsTrainSmall.groupBy(["user_id",'user_id_index',"recording_mbid",'mbid_index']).agg(count("*").alias("play_count"))
    user_total_counts = user_song_counts.groupBy(["user_id",'user_id_index']).agg(count("*").alias("total_count"))
    valid_users = user_total_counts.filter(col("total_count") >= 10).select("user_id")
    valid_user_song_counts = user_song_counts.join(valid_users, on="user_id", how="inner").repartition("user_id")



    valid_user_song_counts.createOrReplaceTempView('valid_user_song_counts')
    
    valid_user_song_ranks=spark.sql('SELECT user_id, user_id_index, recording_mbid, mbid_index,play_count, ROW_NUMBER() over (PARTITION BY user_id ORDER BY play_count DESC) as rank FROM valid_user_song_counts ').createOrReplaceTempView('song_ranks')    
    result=spark.sql('SELECT user_id, user_id_index, recording_mbid, mbid_index, 101-rank as play_count FROM song_ranks as play_count WHERE rank<=100')


    #result = valid_user_song_counts.select("user_id","user_id_index" , "recording_mbid","mbid_index" , "play_count")
    result.write.parquet(f"hdfs:/user/{userID}/user_play_counts_train.parquet") 
    result.show()
    
   
    print('part2')


    
    #Use Previous table to do a train-validation split at user level (So 4 songs they listened to go to Train, others in validation)
    result = spark.read.parquet(f"hdfs:/user/{userID}/user_play_counts_train.parquet",header=True)
    result.createOrReplaceTempView('result')
    user_total_counts = spark.sql('SELECT user_id,user_id_index, count(play_count) as total_count from result group by user_id,user_id_index')
    train_counts = round(user_total_counts["total_count"] * 0.8)
    val_counts = user_total_counts["total_count"] - train_counts
    user_song_splits = result.join(user_total_counts, on="user_id", how="inner") \
        .withColumn("train_count", when(col("play_count") >= train_counts, train_counts).otherwise(col("play_count"))) \
        .withColumn("val_count", when(col("play_count") >= train_counts, col("play_count") - train_counts).otherwise(0)).drop(user_total_counts.user_id_index)

    train_set = user_song_splits.filter(col("train_count") > 0).select("user_id",'user_id_index',"recording_mbid","mbid_index", "train_count")
    validation_set = user_song_splits.filter(col("val_count") > 0).select("user_id","user_id_index", "recording_mbid",'mbid_index', "val_count")

    train_set.write.parquet(f"hdfs:/user/{userID}/train_set.parquet")
    train_set.show()
    validation_set.write.parquet(f"hdfs:/user/{userID}/validation_set.parquet") 
    validation_set.show()
    '''   
    
    #create test set 
    interactionsTest = spark.read.parquet(f'hdfs:/user/{userID}/interactionsTest.parquet',header=True)
    user_song_counts = interactionsTest.groupBy(["user_id",'user_id_index',"recording_mbid",'mbid_index']).agg(count("*").alias("play_count"))
    user_song_counts.createOrReplaceTempView('user_song_counts')
    user_song_ranks=spark.sql('SELECT user_id, user_id_index, recording_mbid, mbid_index,play_count, ROW_NUMBER() over (PARTITION BY user_id ORDER BY play_count DESC) as rank FROM user_song_counts ').createOrReplaceTempView('song_ranks')
    result=spark.sql('SELECT user_id, user_id_index, recording_mbid, mbid_index, 101-rank as play_count FROM song_ranks as play_count WHERE rank<=100')
    result.write.parquet(f"hdfs:/user/{userID}/test_set.parquet") 


if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('indexer').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function
    userID = os.environ['USER']
    main(spark, userID)
