import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD

import math
import joblib

items_raw_df = pd.read_csv('item_properties_part1.csv')
items_raw1_df = pd.read_csv('item_properties_part2.csv')
items_raw_df = pd.concat([items_raw_df, items_raw1_df])
items_raw_df.head()

print(
    f"""
    Có {items_raw_df['itemid'].nunique()} mặt hàng khác nhau.
    Có {items_raw_df['property'].nunique()} thuộc tính khác nhau.
    """)

# Để chạy nhanh hơn, hãy lấy mẫu một số mặt hàng
np.random.seed(1)
items_to_keep = np.random.choice(items_raw_df['itemid'].unique(), size=100000, replace=False)
items_df = items_raw_df[items_raw_df['itemid'].isin(items_to_keep)]

print(
    f"""
    Có {items_df['itemid'].nunique()} mặt hàng khác nhau.
    Có {items_df['property'].nunique()} thuộc tính khác nhau.
    """)
# Để đơn giản, chúng ta sẽ không lấy các thay đổi theo thời gian, thay vào đó chỉ sử dụng giá trị cuối cùng.
items_df = items_df.groupby(['itemid', 'property'])['value'].last().to_frame().reset_index()
items_df.head()
items_df['itemid'].value_counts().head()
events_raw_df = pd.read_csv('events.csv')
events_raw_df.head()
events_df = events_raw_df[events_raw_df['itemid'].isin(items_to_keep)]
events_df = events_df.sort_values(['visitorid', 'timestamp']).reset_index(drop=True)
events_df.head()
print(
    f"""
    Có {events_df['visitorid'].nunique()} người dùng khác nhau trong sự kiện.
    Có {events_df['transactionid'].nunique()} giao dịch khác nhau trong sự kiện.
    Có {events_df['itemid'].nunique()} mặt hàng khác nhau trong sự kiện.
    Các hành động người dùng có thể thực hiện là {events_df['event'].unique()}.
    """
)
print(
    f"""
    {events_df['event'].value_counts()},
    {events_df['event'].value_counts(normalize=True)}
    """
)
print(
    f"""
    Các mặt hàng được mua nhiều nhất: \n {events_df['itemid'].value_counts().head()},
    Người dùng hoạt động nhiều nhất: \n{events_df['visitorid'].value_counts().head()}
    """
)
events_df.dtypes
events_df['itemid'].astype(str).value_counts().head(100).plot.bar()
plt.show()
items_df
# 1 -> "769062 205682 1154859 373 1154859"
items_df = items_df.groupby('itemid')['value'].apply(lambda x: ' '.join(x)).to_frame()
items_df.head()
tfidfvec = TfidfVectorizer(min_df=500, max_df=0.7)
vectorized_data = tfidfvec.fit_transform(items_df['value'])
vectorized_data
tfidf_df = pd.DataFrame(vectorized_data.toarray(),
                       columns=tfidfvec.get_feature_names_out())
tfidf_df.index = items_df.index
cosine_similarity(tfidf_df.loc[3].values.reshape(1, -1),
                  tfidf_df.loc[7].values.reshape(1, -1))
events_df.groupby('visitorid')['itemid'].nunique().sort_values(ascending=False)
seen_items = events_df[events_df['visitorid'] == 1150086]['itemid'].unique()
seen_items[:10]
user_seen_items_df = tfidf_df.reindex(seen_items)
user_profile = user_seen_items_df.mean()
user_profile
user_not_seen_items_df = tfidf_df.drop(seen_items, axis=0)
user_profile_similarities = cosine_similarity(user_profile.values.reshape(1, -1),
                                             user_not_seen_items_df)
user_profile_similarities
user_profile_similarities_df = pd.DataFrame(user_profile_similarities.T,
                                            index=user_not_seen_items_df.index,
                                            columns=['similarity_score'])
user_profile_similarities_df.sort_values('similarity_score', ascending=False).head()
events_df['event'].unique()
visitor_item_event_df = events_df.groupby(['visitorid', 'itemid', 'event'])['timestamp'].count().to_frame().reset_index()
visitor_item_event_df.head()
visitor_item_df = visitor_item_event_df[visitor_item_event_df['event'] == 'addtocart']
visitor_item_df.head()
top_items = visitor_item_df['itemid'].value_counts()[:1000]
top_items
# top_users = visitor_item_df['visitorid'].value_counts()[:500]
visitor_item_df = visitor_item_df[visitor_item_df['itemid'].isin(top_items.index)]
visitor_wise_df = visitor_item_df.pivot(index='visitorid', columns='itemid', values='timestamp')
visitor_wise_df
avg_addcarts = visitor_wise_df.sum() / visitor_wise_df.shape[1]
visitor_wise_raw_df = visitor_wise_df.copy()
visitor_wise_df = visitor_wise_df.sub(avg_addcarts, axis=1)
visitor_wise_df = visitor_wise_df.fillna(0)
visitor_wise_df.head()
item_wise_df = visitor_wise_df.T
item_wise_df
cosine_similarity(item_wise_df.loc[422425].values.reshape(1, -1), 
                  item_wise_df.loc[158666].values.reshape(1, -1))
similarities = cosine_similarity(item_wise_df)
consine_similarity_df = pd.DataFrame(similarities, 
                                    index=item_wise_df.index.values,
                                    columns=item_wise_df.index.values)
consine_similarity_df
consine_similarity_df[56782].sort_values(ascending=False).head(10)
similarities = cosine_similarity(visitor_wise_df)
consine_similarity_df = pd.DataFrame(similarities, 
                                    index=visitor_wise_df.index.values,
                                    columns=visitor_wise_df.index.values)
consine_similarity_df
consine_similarity_df.loc[648].sort_values(ascending=False).head(50)
not_empty = visitor_wise_raw_df.notnull().values.sum()
total_number = visitor_wise_raw_df.size
print(100 * not_empty/total_number)
visitor_wise_sparse = csr_matrix(visitor_wise_df)
U, sigma, Vt = svds(visitor_wise_sparse)

print(U.shape)
print(sigma)
print(np.diag(sigma))
print(Vt.shape)
pd.DataFrame(U, index=visitor_wise_df.index.values)
pd.DataFrame(Vt, columns=visitor_wise_df.columns)
visitor_wise_raw_df
filled_matrix = np.dot(np.dot(U, np.diag(sigma)), Vt)
filled_matrix = filled_matrix + avg_addcarts.values.reshape(1, -1)
pd.DataFrame(filled_matrix,
            index=visitor_wise_raw_df.index,
            columns=visitor_wise_raw_df.columns)
visitor_wise_raw_df
visitor_wise_sparse = csr_matrix(visitor_wise_df)

np.random.seed(42)
mask = np.random.choice([False, True], size = visitor_wise_df.shape, p=[0.2, 0.8])
train_data = np.where(mask, visitor_wise_sparse.toarray(), 0)
test_data = np.where(~mask, visitor_wise_sparse.toarray(), 0)

nonzero_indices = test_data.nonzero()

U, sigma, Vt = svds(train_data)

predicted_matrix = np.dot(np.dot(U, np.diag(sigma)), Vt)
len(nonzero_indices[0])
predicted_test_values = predicted_matrix[nonzero_indices]
actual_test_values = visitor_wise_sparse.toarray()[nonzero_indices]
rmse = math.sqrt(mean_squared_error(predicted_test_values, actual_test_values))
print(rmse)
model = TruncatedSVD()
model.fit(train_data)
# Lưu mô hình SVD
joblib.dump(model, 'svd_model.pkl')

# Tải mô hình SVD
loaded_model = joblib.load('svd_model.pkl')