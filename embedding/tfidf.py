# 前処理用
import re
import os
import pandas as pd
import numpy as np
import emoji
import spacy
import neologdn
import json

# 分かち書き用
import ginza
import ja_ginza_electra
# pandas高速化
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

print("Load data!")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
def wakati_rm_func(x):
    if x is not np.nan:
        nlp = spacy.load('ja_ginza_electra')
        sentence = x
        
        sentence = re.sub(r'[!-~]'," ",sentence) # 小文字の記号を削除
        sentence=re.sub(r'[︰-＠]', "", sentence) # 大文字の記号を削除
        # 絵文字削除
        sentence = ''.join(['' if c in emoji.UNICODE_EMOJI else c for c in sentence])

        # 不要記号削除
        pattern = '[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”◇ᴗ●↓→♪★⊂⊃※△□◎〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％�]'
        sentence =  re.sub(pattern, ' ', sentence)

        # 正規化する
        sentence = neologdn.normalize(sentence)
    
        # 大文字・小文字変換
        sentence = sentence.lower()
        sentence = sentence.replace("\n", "")
        # GinZaで分かち書きをする
        doc = nlp(sentence)
        nlp = spacy.load('ja_ginza_electra')
        tmp_words_list = []
        for sent in doc.sents:
            for token in sent:
                if token.pos_ in ["PROPN","NOUN", "ADJ", "VERB"]:
                    tmp_words_list.append(token.orth_)

        result = " ".join(tmp_words_list)
        return result
    else:
        return " "

# trainに分かち書きを実行する
train["story_wakati"] = train[["story"]].parallel_apply(wakati_rm_func)
train["title_wakati"] = train[["title"]].parallel_apply(wakati_rm_func)
train["keyword_wakati"] = train[["keyword"]].parallel_apply(wakati_rm_func)

# testに分かち書きを実行する
test["story_wakati"] = test[["story"]].parallel_apply(wakati_rm_func)
test["title_wakati"] = test[["title"]].parallel_apply(wakati_rm_func)
test["keyword_wakati"] = test[["keyword"]].parallel_apply(wakati_rm_func)

# Countvectorizeする
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train["story_wakati"])
count_vec_story_train = pd.DataFrame(data= X.toarray(), columns = vectorizer.get_feature_names())

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train["title_wakati"])
count_vec_title_train = pd.DataFrame(data= X.toarray(), columns = vectorizer.get_feature_names())

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train["keyword_wakati"])
count_vec_keyword_train = pd.DataFrame(data= X.toarray(), columns = vectorizer.get_feature_names())

pd.concat([count_vec_story_train, count_vec_title_train, count_vec_keyword_train]).to_pickle("data/train_countvec.pkl")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(test["story_wakati"])
count_vec_story_test = pd.DataFrame(data= X.toarray(), columns = vectorizer.get_feature_names())

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(test["title_wakati"])
count_vec_title_test = pd.DataFrame(data= X.toarray(), columns = vectorizer.get_feature_names())

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(test["keyword_wakati"])
count_vec_keyword_test = pd.DataFrame(data= X.toarray(), columns = vectorizer.get_feature_names())

pd.concat([count_vec_story_test, count_vec_title_test, count_vec_keyword_test]).to_pickle("data/test_countvec.pkl")

# TI-IDFを計算する
from sklearn.feature_extraction.text import TfidfVectorizer
model = TfidfVectorizer(max_df=0.9)
X = model.fit_transform(train["story_wakati"])
story_tfidf_train = pd.DataFrame(data= X.toarray(), columns = model.get_feature_names())

model = TfidfVectorizer(max_df=0.9)
X = model.fit_transform(train["title_wakati"])
title_tfidf_train = pd.DataFrame(data= X.toarray(), columns = model.get_feature_names())

model = TfidfVectorizer(max_df=0.9)
X = model.fit_transform(train["keyword_wakati"])
keyword_tfidf_train = pd.DataFrame(data= X.toarray(), columns = model.get_feature_names())

pd.concat([story_tfidf_train, title_tfidf_train, keyword_tfidf_train]).to_pickle("data/train_tfidf.pkl")

model = TfidfVectorizer(max_df=0.9)
X = model.fit_transform(test["story_wakati"])
story_tfidf_test = pd.DataFrame(data= X.toarray(), columns = model.get_feature_names())

model = TfidfVectorizer(max_df=0.9)
X = model.fit_transform(test["title_wakati"])
title_tfidf_test = pd.DataFrame(data= X.toarray(), columns = model.get_feature_names())

model = TfidfVectorizer(max_df=0.9)
X = model.fit_transform(test["keyword_wakati"])
keyword_tfidf_test = pd.DataFrame(data= X.toarray(), columns = model.get_feature_names())

pd.concat([story_tfidf_test, title_tfidf_test, keyword_tfidf_test]).to_pickle("data/test_tfidf.pkl")