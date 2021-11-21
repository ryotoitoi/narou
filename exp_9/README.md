# LightGBM＋RoBERTa＋Universal Sentence Encoderを試す。
- 前処理は`preprocessing.ipynb`でやる。
- 学習を`train.py`にて行う。

## 前処理について
- `/npy`から.npyのファイルをロードする必要がある。
- `exp_{num}/data`に新しく作成したデータを保存する必要があり。

## 学習について
- `LightGBM`を使用する。
- `optuna`によってハイパーパラメータをチューニングする。
