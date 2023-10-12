import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, ntile, split, count, desc, round, sum, when, concat_ws
from pyspark.sql.window import Window
from pyspark.sql.functions import coalesce


def main(spark):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object
    '''
    #####--------------YOUR CODE STARTS HERE--------------#####
    
    
    tracksTest = spark.read.parquet('hdfs:///user/bm106_nyu_edu/1004-project-2023/tracks_test.parquet', header=True)
    tracksTest.createOrReplaceTempView('tracksTest')

    usersTest = spark.read.parquet('hdfs:///user/bm106_nyu_edu/1004-project-2023/users_test.parquet', header=True)
    usersTest.createOrReplaceTempView('usersTest')
    

    interactionsTest = spark.read.parquet('hdfs:///user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet', header=True)
    #create track artist key 
    interactionsTest = interactionsTest.join(
    tracksTest.select("recording_msid", "recording_mbid"),
    on="recording_msid",
    how="left"
)
    interactionsTest =  interactionsTest.withColumn("recording_mbid", coalesce("recording_mbid", "recording_msid"))
    ''' 
    #Create Time Based Train-Validation split
    interactionsTest = interactionsTest.orderBy(col("timestamp"))
    total_rows = interactionsTest.count()
    split_row1 = int(total_rows * 0.4)
    split_row2 = int(total_rows * 0.9)
    
    w = Window.orderBy(col("timestamp"))
    interactionsTrain = interactionsTrain.withColumn("tile", ntile(10).over(w))

    train_set_time = interactionsTrain.filter((col("tile") <= 4) | (col("tile") >= 9)).drop("tile")
    validation_set_time = interactionsTrain.filter((col("tile") >= 5) & (col("tile") <= 8)).drop("tile")

    train_set_time.write.parquet("hdfs:/user/ra2829_nyu_edu/train_set_time.parquet") 
    validation_set_time.write.parquet("hdfs:/user/ra2829_nyu_edu/validation_set_time.parquet") 
    '''
    
    
    
    #Get frequency table at user-song level. Eliminate all users with less than 5 songs listened (as we are doing a 80-20 split)
    user_song_counts = interactionsTest.groupBy(["user_id","recording_mbid"]).agg(count("*").alias("play_count"))
    user_total_counts = user_song_counts.groupBy("user_id").agg(count("*").alias("total_count"))
    valid_users = user_total_counts.filter(col("total_count") >= 10).select("user_id")
    valid_user_song_counts = user_song_counts.join(valid_users, on="user_id", how="inner")
    result = valid_user_song_counts.select("user_id", "recording_mbid", "play_count")
    result.write.parquet("hdfs:/user/ra2829_nyu_edu/user_play_counts_test.parquet") 
    result.show()
    
    
    '''
    #Use Previous table to do a train-validation split at user level (So 4 songs they listened to go to Train, others in validation)
    result = spark.read.parquet("hdfs:/user/ra2829_nyu_edu/user_play_counts_test.parquet",header=True)
    result.createOrReplaceTempView('result')
    user_total_counts = spark.sql('SELECT user_id, count(play_count) as total_count from result group by user_id')
    train_counts = round(user_total_counts["total_count"] * 0.8)
    val_counts = user_total_counts["total_count"] - train_counts
    user_song_splits = result.join(user_total_counts, on="user_id", how="inner") \
        .withColumn("train_count", when(col("play_count") >= train_counts, train_counts).otherwise(col("play_count"))) \
        .withColumn("val_count", when(col("play_count") >= train_counts, col("play_count") - train_counts).otherwise(0))

    train_set = user_song_splits.filter(col("train_count") > 0).select("user_id","recording_mbid", "train_count")
    validation_set = user_song_splits.filter(col("val_count") > 0).select("user_id", "recording_mbid", "val_count")

    train_set.write.parquet("hdfs:/user/ra2829_nyu_edu/train_set.parquet") 
    validation_set.write.parquet("hdfs:/user/ra2829_nyu_edu/validation_set.parquet") 
    
    
    #Use the train set, to get the popular 100 songs for the popularity baseline model
    test_set = spark.read.parquet("hdfs:/user/ra2829_nyu_edu/test_set.parquet", header=True)
    test_set.createOrReplaceTempView('test_set')
    song_counts = test_set.groupBy("recording_mbid").agg(sum("train_count").alias("pop_count"))
    top_songs = song_counts.orderBy(desc("pop_count")).limit(100)
    top_songs.write.parquet("hdfs:/user/ra2829_nyu_edu/popular_train.parquet")
    top_songs.show()
    ''' 

    

        
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part2').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
    print('doneeeee!')
