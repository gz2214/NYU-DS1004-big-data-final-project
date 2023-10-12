import sys
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, ntile, split, count, desc, round, sum, when, concat_ws,dense_rank
from pyspark.sql.window import Window
from pyspark.sql.functions import coalesce
from pyspark.ml.feature import StringIndexer

def main(spark, userID):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object
    '''
    #####--------------YOUR CODE STARTS HERE--------------#####
    
    
    tracksTrainSmall = spark.read.parquet('hdfs:///user/bm106_nyu_edu/1004-project-2023/tracks_train.parquet', header=True)
    tracksTrainSmall.createOrReplaceTempView('tracksTrainSmall')
    tracksTest=spark.read.parquet('hdfs:///user/bm106_nyu_edu/1004-project-2023/tracks_test.parquet', header=True)

    print('users index')
    usersTrainSmall = spark.read.parquet('hdfs:///user/bm106_nyu_edu/1004-project-2023/users_train.parquet', header=True)
    usersTrainSmall.createOrReplaceTempView('usersTrainSmall')

    usersTest=spark.read.parquet('hdfs:///user/bm106_nyu_edu/1004-project-2023/users_test.parquet', header=True)
    usersTest.createOrReplaceTempView('usersTest')
    
    usersAll=spark.sql('SELECT user_id FROM usersTrainSmall UNION SELECT user_id FROM usersTest')
    usersAll.createOrReplaceTempView('usersAll')
    usersIndex=spark.sql('SELECT user_id, row_number() over (order by user_id)  user_id_index FROM usersAll').createOrReplaceTempView('usersIndex')

    print('interactionsIndex')
    interactionsTrainSmall = spark.read.parquet('hdfs:///user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet', header=True)
    interactionsTest=spark.read.parquet('hdfs:///user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet', header=True)
    
    print('#coalesce msid and mbid')
    interactionsTrainSmall = interactionsTrainSmall.join(
    tracksTrainSmall.select("recording_msid", "recording_mbid"),
    on="recording_msid",
    how="left") 
    interactionsTrainSmall =  interactionsTrainSmall.withColumn("recording_mbid", coalesce("recording_mbid", "recording_msid"))
    interactionsTrainSmall.createOrReplaceTempView('interactionsTrain')
    
    interactionsTest=interactionsTest.join(tracksTest.select('recording_msid','recording_mbid'),on='recording_msid',how='left')
    interactionsTest=interactionsTest.withColumn('recording_mbid',coalesce('recording_mbid','recording_msid'))
    interactionsTest.createOrReplaceTempView('interactionsTest')
    
    interactionsAll=spark.sql('SELECT recording_mbid FROM interactionsTrain UNION SELECT recording_mbid FROM interactionsTest')
    interactionsAll.createOrReplaceTempView('interactionsAll')
    interactionsIndex=spark.sql('SELECT recording_mbid, row_number() over (order by recording_mbid)  mbid_index FROM interactionsAll').createOrReplaceTempView('mbidIndex')
    
    '''    
    print('join index')
    interactionsTrainSmall=spark.sql('SELECT i.user_id,u.user_id_index,i.timestamp,i.recording_mbid,m.mbid_index FROM interactionsTrain i\
            LEFT JOIN usersIndex u on i.user_id=u.user_id\
            LEFT JOIN mbidIndex m on i.recording_mbid=m.recording_mbid')
    print('for train')
    interactionsTrainSmall.repartition("user_id").write.parquet(f'hdfs:/user/{userID}/interactionsTrain.parquet')
    interactionsTrainSmall.show()
    '''

    print('now for test FINALLY YAAAAAAAA')
    interactionsTest=spark.sql('SELECT i.user_id,u.user_id_index,i.timestamp,i.recording_mbid,m.mbid_index FROM interactionsTest i\
            LEFT JOIN usersIndex u on i.user_id=u.user_id\
            LEFT JOIN mbidIndex m on i.recording_mbid=m.recording_mbid')
    interactionsTest.show()
    interactionsTest.write.parquet(f'hdfs:/user/{userID}/interactionsTest.parquet')
            
    
    
    '''
    user_index=StringIndexer(inputCol="user_id", outputCol="user_id_index",handleInvalid="keep")
    unique_mbid=interactionsTrainSmall.select("recording_mbid").distinct()
    window = Window.orderBy("recording_mbid")
    index_mbid=unique_mbid.withColumn("mbid_index", dense_rank().over(window) - 1)
    interactionsTrainSmall=user_index.fit(interactionsTrainSmall).transform(interactionsTrainSmall)
    interactionsTrainSmall=interactionsTrainSmall.join(index_mbid,on='recording_mbid',how='inner')
    interactionsTrainSmall.write.parquet(f'hdfs:/user/{userID}/interactionsTrain.parquet')
    interactionsTrainSmall.show()
    '''

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('indexer').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function
    userID = os.environ['USER']
    main(spark, userID)
