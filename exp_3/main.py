import json
import pickle
import re
from glob import glob
from tqdm import tqdm

import regex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

import spacy
nlp = spacy.load('ja_ginza')

from sklearn.metrics import log_loss
from xfeat import Pipeline, SelectCategorical, LabelEncoder

### ファイル読み込み・データ確認


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
sub_df = pd.read_csv('data/sample_submission.csv')

raw_df = pd.concat([train_df, test_df])
train_idx = train_df.shape[0] # 何行目までが学習データか、後ほど使う
print(raw_df.shape)
raw_df.head(2)

### feature engineering


import datetime

dt_now = datetime.datetime.now()
raw_df['past_days'] = raw_df['general_firstup'].apply(lambda x: (dt_now - datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)

encoder = Pipeline([
    SelectCategorical(),
    LabelEncoder(output_suffix=""),
])

le_df = encoder.fit_transform(raw_df)
le_df.head(2)

raw_df['writer'] = le_df['writer']

#### テキストから古典的な特徴量作成


def create_pos_features(texts):
    """品詞ベースの特徴量を作成"""

    docs = list(tqdm(nlp.pipe(texts, disable=['ner'])))

    pos_data = {}

    negative_auxs = [0]*len(docs)

    for i, doc in enumerate(docs):
        for token in doc:
            # 品詞が○○である形態素の数。後で全形態素数で割る
            if token.pos_ not in pos_data:
                pos_data[token.pos_] = [0]*len(docs)
            pos_data[token.pos_][i] += 1

            # 品詞が助動詞の「ない」「ぬ」「ん」の数。後で全形態素数で割る
            if token.pos_ == 'AUX' and token.lemma_ in ['ない', 'ぬ', 'ん']:
                negative_auxs[i] += 1

    pos_df = pd.DataFrame.from_dict(pos_data, orient='index').T
    pos_df['num_token'] = pos_df.sum(axis=1) # 全形態素数

    pos_df['NEG_AUX'] = negative_auxs

    for colname in pos_df.columns:
        if colname != 'num_token':
            pos_df[colname] /= pos_df['num_token'] # 全形態素数で割る

    return pos_df


def create_type_features(texts):
    """文字種ベースの特徴量を作成"""

    type_data = []

    for text in texts:
        tmp = []
        
        tmp.append(len(text))

        # 平仮名の文字数カウント
        p = re.compile('[\u3041-\u309F]+')
        s = ''.join(p.findall(text))
        tmp.append(len(s))

        # カタカナの文字数カウント
        p = re.compile('[\u30A1-\u30FF]+')
        s = ''.join(p.findall(text))
        tmp.append(len(s))

        # 漢字の文字数カウント
        p = regex.compile(r'\p{Script=Han}+')
        s = ''.join(p.findall(text))
        tmp.append(len(s))

        # 絵文字の文字数カウント
        p = regex.compile(r'\p{Emoji_Presentation=Yes}+')
        s = ''.join(p.findall(text))
        tmp.append(len(s))

        type_data.append(tmp)

    colnames = ['length', 'hiragana_length', 'katakana_length', 'kanji_length', 'emoji_length']
    type_df = pd.DataFrame(type_data, columns=colnames)

    for colname in type_df.columns:
        if colname != 'length':
            type_df[colname] /= type_df['length']

    return type_df

# まずタイトルに対して処理

titles = raw_df['title'].values
docs = list(nlp.pipe(titles, disable=['ner']))

title_pos_df = create_pos_features(titles)
title_pos_df.columns = ['title_' + colname for colname in title_pos_df.columns]

title_type_df = create_type_features(titles)
title_type_df.columns = ['title_' + colname for colname in title_type_df.columns]

# 次にあらすじに対して処理

stories = raw_df['story'].values

# かなり大きなメモリを使用するので、念の為データを分割しバッチ処理

nrow_one_loop = 20000
nloop = np.floor(len(stories)/nrow_one_loop)
min_idx = 0

story_pos_dfs = []

while min_idx < len(stories):
    tmp_stories = stories[min_idx:min_idx+nrow_one_loop]
    suffix = str(min_idx//nrow_one_loop) if len(str(min_idx//nrow_one_loop)) != 1 else '0'+str(min_idx//nrow_one_loop)
    story_pos_dfs.append(create_pos_features(tmp_stories))
    min_idx += nrow_one_loop

story_pos_df = pd.concat(story_pos_dfs)
del story_pos_dfs

# かなり大きなメモリを使用するので、念の為データを分割しバッチ処理

nrow_one_loop = 20000
nloop = np.floor(len(stories)/nrow_one_loop)
min_idx = 0

story_type_dfs = []

while min_idx < len(stories):
    tmp_stories = stories[min_idx:min_idx+nrow_one_loop]
    story_type_dfs.append(create_type_features(tmp_stories))
    min_idx += nrow_one_loop

story_type_df = pd.concat(story_type_dfs)
del story_type_dfs

#### BERTVectorizer

# 上では古典的な特徴量作成を行いましたが、より近代的な方法も試してみましょう。<br>
# BERTを使って、文章をベクトル表現に変換してみます。


import pandas as pd
import numpy as np
import torch
import transformers

from transformers import BertJapaneseTokenizer
from tqdm import tqdm
tqdm.pandas()

class BertSequenceVectorizer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 128
            

    def vectorize(self, sentence : str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)
        
        seq_out = self.bert_model(inputs_tensor, masks_tensor)[0]
        pooled_out = self.bert_model(inputs_tensor, masks_tensor)[1]

        if torch.cuda.is_available():    
            return seq_out[0][0].cpu().detach().numpy() # 0番目は [CLS] token, 768 dim の文章特徴量
        else:
            return seq_out[0][0].detach().numpy()

BSV = BertSequenceVectorizer()
raw_df['title_feature'] = raw_df['title'].progress_apply(lambda x: BSV.vectorize(x) if x is not np.nan else np.array([0]*768))

title_bert_df = pd.DataFrame(raw_df['title_feature'].tolist())
title_bert_df.columns = ['title_bertvec_'+str(col) for col in title_bert_df.columns]

BSV = BertSequenceVectorizer()
raw_df['story_feature'] = raw_df['story'].progress_apply(lambda x: BSV.vectorize(x) if x is not np.nan else np.array([0]*768))

story_bert_df = pd.DataFrame(raw_df['story_feature'].tolist())
story_bert_df.columns = ['story_bertvec_'+str(col) for col in story_bert_df.columns]

### DataFrame結合

raw_df.reset_index(drop=True, inplace=True)
title_pos_df.reset_index(drop=True, inplace=True)
title_type_df.reset_index(drop=True, inplace=True)
story_pos_df.reset_index(drop=True, inplace=True)
story_type_df.reset_index(drop=True, inplace=True)
title_bert_df.reset_index(drop=True, inplace=True)
story_bert_df.reset_index(drop=True, inplace=True)

concat_df = pd.concat([raw_df, title_pos_df, title_type_df, story_pos_df, story_type_df, title_bert_df, story_bert_df], axis=1)
concat_df.shape

### データ分割、学習、評価

# LightGBMで学習、評価します。バリデーションはHold-out法を採用しています

# カテゴリ変数・連続変数各々のカラムを特定

cat_cols = ['userid', 'writer', 'biggenre', 'genre', 'novel_type', 'end', 'isstop', 'isr15', 'isbl', 'isgl', 'iszankoku', 'istensei', 'istenni', 'pc_or_k']

num_cols = ['past_days']
num_cols += list(title_pos_df.columns) + list(title_type_df.columns) + list(story_pos_df.columns) + list(story_type_df.columns) + list(title_bert_df.columns) + list(story_bert_df.columns)

feat_cols = cat_cols + num_cols

ID = 'ncode'
TARGET = 'fav_novel_cnt_bin'

# バリデーションはHold-out法（一定割合で学習データと評価データの2つに分割）で行う

train_df = concat_df.iloc[:35000, :]
val_df = concat_df.iloc[35000:train_idx, :]
test_df = concat_df.iloc[train_idx:, :]
print(train_df.shape, val_df.shape, test_df.shape)

train_x = train_df[feat_cols]
train_y = train_df[TARGET]
val_x = val_df[feat_cols]
val_y = val_df[TARGET]
test_x = test_df[feat_cols]
test_y = test_df[TARGET]
train_x.shape

# 評価指標はMulti-class logloss

SEED = 0

params = {
    'objective': 'multiclass',
    'num_classes': 5,
    'metric': 'multi_logloss',
    'num_leaves': 42,
    'max_depth': 7,
    "feature_fraction": 0.8,
    'subsample_freq': 1,
    "bagging_fraction": 0.95,
    'min_data_in_leaf': 2,
    'learning_rate': 0.1,
    "boosting": "gbdt",
    "lambda_l1": 0.1,
    "lambda_l2": 10,
    "verbosity": -1,
    "random_state": 42,
    "num_boost_round": 50000,
    "early_stopping_rounds": 100
}

train_data = lgb.Dataset(train_x, label=train_y)
val_data = lgb.Dataset(val_x, label=val_y)

model = lgb.train(
    params,
    train_data, 
    categorical_feature = cat_cols,
    valid_names = ['train', 'valid'],
    valid_sets =[train_data, val_data], 
    verbose_eval = 100,
)

val_pred = model.predict(val_x, num_iteration=model.best_iteration)

pred_df = pd.DataFrame(sorted(zip(val_x.index, val_pred, val_y)), columns=['index', 'predict', 'actual'])

feature_imp = pd.DataFrame(sorted(zip(model.feature_importance(), train_x.columns)), columns=['importance', 'feature'])

# 特徴量の重要度を確認

lgb.plot_importance(model, figsize=(12,8), max_num_features=50, importance_type='gain')
plt.tight_layout()
plt.show()

# 評価指標はlog lossだが、accuracyも見てみる

val_pred = model.predict(val_x, num_iteration=model.best_iteration)
val_pred_max = np.argmax(val_pred, axis=1)  # 最尤と判断したクラスの値にする
accuracy = sum(val_y == val_pred_max) / len(val_y)
print(accuracy)

### 推論・投稿ファイル作成

test_pred = model.predict(test_x, num_iteration=model.best_iteration)

sub_df.iloc[:, 1:] = test_pred

sub_df.to_csv('./output/test_submission_exp_3.csv', index=False)

