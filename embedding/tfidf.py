# TF-IDFして次元圧縮をする
import pandas as pd
train = pd.read_pickle("./data/wakati_train.pkl")
test = pd.read_pickle("./data/wakati_test.pkl")
df = pd.concat([train, test])
df.head()
# TI-IDFを計算する
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

text_col = "story"
for text_col in ["story", "title", "keyword"]:
    print(f"Start {text_col}")
    model = TfidfVectorizer()
    tfidf = model.fit_transform(df[text_col + "_wakati"])
    print(tfidf.shape)
    # 次元圧縮
    svd = TruncatedSVD(64)
    svd.fit(tfidf)
    tfidf_svd = svd.transform(tfidf)
    print(tfidf_svd.shape)

    tfidf_svd_train = tfidf_svd[:40000]
    tfidf_svd_test = tfidf_svd[40000:]
    train_df = pd.DataFrame(tfidf_svd_train)
    test_df = pd.DataFrame(tfidf_svd_test)

    for col_name in train_df.columns:
        train_df = train_df.rename(
            columns={col_name: f"tfidf_{text_col}_{col_name}"})

    for col_name in test_df.columns:
        test_df = test_df.rename(
            columns={col_name: f"tfidf_{text_col}_{col_name}"})

    train_df.to_csv(f"./tfidf/tfidf_{text_col}_train.csv", index = False)
    test_df.to_csv(f"./tfidf/tfidf_{text_col}_test.csv", index = False)



