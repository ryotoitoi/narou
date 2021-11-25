import pandas as pd
train = pd.read_pickle("./data/wakati_train.pkl")
test = pd.read_pickle("./data/wakati_test.pkl")

df = pd.concat([train, test])

del train
del test

# TI-IDFを計算する
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np


model = TfidfVectorizer()
X = model.fit_transform(df["story_wakati"])
story_tfidf= pd.DataFrame(data= X.toarray(), columns = model.get_feature_names())
print(story_tfidf)
print(story_tfidf.shape)

# 次元圧縮
print("次元圧縮を始めます！")
svd = TruncatedSVD(64)
svd.fit(story_tfidf)

# batch処理
nrow_one_loop = 1000
nloop = np.floor(len(story_tfidf)/nrow_one_loop)
min_idx = 0

story_pos_dfs = []
while min_idx < len(story_tfidf):
    tmp_stories = story_tfidf[min_idx:min_idx+nrow_one_loop]
    X = svd.transform(tmp_stories)
    story_pos_dfs.append(pd.DataFrame(X))
    min_idx += nrow_one_loop

story_tfidf_svd = pd.concat(story_pos_dfs)
del story_pos_dfs
print(story_tfidf_svd.shape)

story_tfidf_train  = pd.DataFrame(story_tfidf_svd[:40000])
story_tfidf_test  = pd.DataFrame(story_tfidf_svd[40000:])
for col_name in story_tfidf_train.columns:
    story_tfidf_train = story_tfidf_train.rename(
        columns={col_name: f"title_{col_name}"})

for col_name in story_tfidf_test.columns:
    story_tfidf_test = story_tfidf_test.rename(
        columns={col_name: f"title_{col_name}"})

print("train shape",story_tfidf_train.shape)
print("test shape",story_tfidf_test.shape)