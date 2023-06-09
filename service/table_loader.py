from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://user:password@host:dbname"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


# Загрузка таблиц с признаками для обучения модели из базы данных
posts = batch_load_sql("""SELECT *  FROM "s_gagarin_posts_feats" """)
posts = posts.drop('index', axis=1)
posts_new = posts[['post_id']] # для формирования рейтинга постов в функции get_top_posts

users = batch_load_sql("""SELECT *  FROM "s_gagarin_users_feats" """)
users = users.drop('index', axis=1)
users = users.set_index('user_id')

feed_liked = batch_load_sql("""SELECT *
                            FROM feed_data
                            WHERE target = 1
                            """)

# Загрузка модели
xgboost = pickle.load(open('xgb_model_auc_069.pkl', 'rb'))


def get_df_for_predict(user_id: int, time) -> np.array:
    """Функция для пролучения массива для предсказаний по искомому пользователю"""

    user = users.loc[user_id].values
    user_matx = np.repeat([user], posts.shape[0], axis=0)

    return np.hstack((user_matx, posts.drop('post_id', axis=1).values))


def get_top_posts(user_id, time, limit) -> np.array:
    """Формирование рейтинга постов по предсказанной вероятности"""

    posts_new['pred_proba'] = xgboost.predict_proba(get_df_for_predict(user_id, time))[:, 1]
    posts_liked = feed_liked[(feed_liked['user_id'] == user_id) & (feed_liked['timestamp'] < time)]['post_id'].values # отбираем посты по времении, которые уже были ранее
    posts_new_filtered = posts_new[~posts_new['post_id'].isin(posts_liked)] # Убираем из предсказания посты, которые уже были лайкнуты
    posts_new_filtered.sort_values('pred_proba', ascending=False, inplace=True)

    return posts_new_filtered.head(limit)['post_id'].values