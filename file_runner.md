# How to run the files (for popularity baseline and ALS):
1. get_index.py: This gets the indices for recording_mbid and user_id. This outputs a user counts file.
2. train-val-split.py: This performs the train-val split. It takes a user counts file. 
3. get_popular_songs.py: This gets the 100 most popular songs (only needed for popularity baseline)
4. recommendation_table.py: This makes the recommednation table in the user id and recommendations and ground truth in the list format.
5. evaluation.py: This gets the MAP estimate for popularity baseline model. Prints out the MAP for training and validation datasets.
6. als-evaluation.py: This creates the recommendation table using the ALS algorithm. This is where hyperparamter tuning needs to be done for ALS model.
7. als-evaluation-metric.py: This gets the MAP estimate for the ALS model. Prints out the MAP for training and validation datasets.

Run all the files in the order specified above.
