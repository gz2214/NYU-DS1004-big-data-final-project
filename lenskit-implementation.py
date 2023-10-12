from lenskit.algorithms import Recommender
from lenskit.algorithms.als import ImplicitMF
import pandas as pd
from lenskit import batch

train_path = 'train_set.parquet'

train_set = dd.read_parquet(train_path).compute()

train_df1 = train_set.drop('user_id',axis=1)
train_df = train_df1.drop('recording_mbid',axis=1)
train_df['train_count'] = train_df['train_count'].astype(int)
train_df = train_df.rename(columns={'user_id_index': 'user', 'mbid_index': 'item', 'train_count': 'rating'})
als = ImplicitMF(features=8, iterations=10, reg=0.1, weight=10, use_ratings=True)

als.fit(train_df)
recommender = Recommender.adapt(als)

recommendations = batch.recommend(recommender, train_df.user.unique(), 100)

recommendations_by_user = {}
for i, row in enumerate(data['user']):
    user = data['user'][i]
    recommendation = data['recommendation'][i]
    if user not in recommendations_by_user:
        recommendations_by_user[user] = []
    recommendations_by_user[user].append(recommendation)

result = []
for user, recommendations in recommendations_by_user.items():
    result.append((user, recommendations))

recommendation_df = pd.DataFrame(result, columns=['user_id_index', 'recommendations'])

#Since we are not inpyspark framework, we use a user defined function found through stack overflow to evaluate map scores.
def apk(actual, predicted, k=100):
        if len(predicted)>k:
            predicted = predicted[:k]
        score = 0.0
        num_hits = 0.0
        for i,p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)
        if not actual:
            return 0.0
        return score / min(len(actual), k)
