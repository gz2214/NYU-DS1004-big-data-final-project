from pyspark.sql.functions import col,sum,collect_list
import sys
import os
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import BucketedRandomProjectionLSH,Normalizer
from pyspark.sql import functions as F
# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
def main(spark, userID):
    
    train=spark.read.parquet(f"hdfs:/user/{userID}/train_set.parquet")
    
    user_songs_df = train.groupBy("user_id_index").agg(collect_list("recording_mbid").alias("mbid_ids"))
    cv = CountVectorizer(inputCol="mbid_ids", outputCol="features", binary=True)
    matrix = cv.fit(user_songs_df).transform(user_songs_df).select("user_id_index", "features")
    # Create Normalizer transformer
    normalizer = Normalizer(inputCol="features", outputCol="norm_features")

    # Fit and transform the binary matrix to get normalized vectors
    norm_matrix = normalizer.transform(matrix).select("user_id_index", "norm_features")
    brp = BucketedRandomProjectionLSH(inputCol="norm_features", outputCol="hashes", bucketLength=1.0, numHashTables=5)
    norm_matrix.show()
    # fit the model to the transformed data
    print('fitting model')
    model = brp.fit(norm_matrix)



    

    

    similar_users=model.approxSimilarityJoin(norm_matrix,norm_matrix,.5,"cosineSimilarity")
    similar_users = similar_users.select(col("datasetA.user_id_index").alias("user_id_index"),col("datasetB.user_id_index").alias("user_id_index1"),col("ManhattanDistance"))
    
    
    print('writing')
    similar_users.write.parquet(f'hdfs:/user/{userID}/similar_users.parquet')
    print('done')
    #similar_users.show()
    
    
    similar_users=spark.read.parquet(f'hdfs:/user/{userID}/similar_users.parquet')
    similar_users.filter(col('user_id_index1')!=col('user_id_index'))
    train.createOrReplaceTempView('train')
    similar_users.createOrReplaceTempView('similar')
    similar_songs0=spark.sql('SELECT s.user_id_index,t.mbid_index,t.train_count FROM similar s inner join train t on t.user_id_index=s.user_id_index1')
    #do not recommend songs they already listened to 
    similar_songs0.createOrReplaceTempView('similar_songs0')
    similar_songs=spark.sql('SELECT DISTINCT s.user_id_index,s.mbid_index, s.train_count FROM similar_songs0 s inner join train t on t.user_id_index=s.user_id_index WHERE s.mbid_index!=t.mbid_index')
    
    print('creating scores')
    similar_songs.createOrReplaceTempView('similar_songs')
    similar_songs_agg=spark.sql('SELECT user_id_index, mbid_index, SUM(train_count) train_count FROM similar_songs GROUP BY user_id_index, mbid_index')
    similar_songs.createOrReplaceTempView('similar_songs_agg')
    similar_songs_scores=spark.sql('SELECT user_id_index,mbid_index,ROW_NUMBER() over (PARTITION BY user_id_index ORDER BY train_count DESC) as rank FROM similar_songs_agg')
    similar_songs_scores.createOrReplaceTempView('similar_songs_scores')
    similar_recs=spark.sql('SELECT user_id_index,mbid_index FROM similar_songs_scores WHERE Rank<=100')
    
    print('collecting')
    recommendation_table=similar_recs.groupBy("user_id_index").agg(F.collect_list("mbid_index").alias("recommendations_nn"))
    
    print('writing')
    recommendation_table.write.parquet(f'hdfs:/user/{userID}/recommendations_nn.parquet')
    print('plz')
    recommendation_table.show()
    
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('annoy').config("spark.sql.pivotMaxValues", "1500000").getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function
    userID = os.environ['USER']
    main(spark, userID)
    print('done omg')

