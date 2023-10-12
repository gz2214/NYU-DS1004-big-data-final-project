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
    train_set = spark.read.parquet(f"hdfs:/user/{userID}/train_set_small.parquet", header=True)
    train_set.createOrReplaceTempView('train_set')

    song_scores=spark.sql("SELECT recording_mbid, mbid_index, sum(train_count) sumScores, COUNT(DISTINCT user_id)+10000 numUser FROM train_set GROUP BY recording_mbid, mbid_index ").createOrReplaceTempView('song_scores')
    top_songs=spark.sql('SELECT recording_mbid, mbid_index, sumScores/numUser as pop_count FROM song_scores').orderBy(desc("pop_count")).limit(100)


    '''
    song_counts = train_set.groupBy(["recording_mbid",'mbid_index']).agg(sum("train_count").alias("pop_count"))
    top_songs = song_counts.orderBy(desc("pop_count")).limit(100)
    '''
    
    top_songs.write.parquet(f"hdfs:/user/{userID}/popular_train_small.parquet")
    top_songs.show()
    

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('indexer').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function
    userID = os.environ['USER']
    main(spark, userID)
