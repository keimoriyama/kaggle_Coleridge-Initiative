# Coleridge Initiative - Show US the Data

## Dicovery

dataset_titleとdataset_labelは必ずしも一致するわけではない

訓練用データに含まれるデータセットの数は45種類

## Qeustions

dataset_titleとdataset_labelの違いは？

名詞と動詞とかだけ（冠詞とか前置詞はいらないんじゃね？）を除いたらどうなるかな？

## Description

NLPを使って引用されたデータセットを特定する。

語とデータセットの関係をみつけたい。

論文の中で使われたデーセットの特定

## Evaluation

Jaccard-based FBetaにより評価。評価に使われるコードは公開されている。

Beta≒0.5以上（？）のものを対象にして評価し、スコアを計算する。

複数のデータセットが使われたと判断された場合には、`|`でつないで提出する。

### 注意点

- データセットの名前はアルファベット順にしないとだめ。

- TP,FP,FNが全ての提出ファイルに対して計算され、microF0.5スコアを計算する。

## Data

Json形式で配布されている。セクション毎に区切られている。

### Files

    - train:学習データ
    - test:テストデータ

### Columns of trian.csv

- id

論文id

- pub_title

論文のタイトル

- dataset_title

データセットの名前

- dataset_label

データセットを表すテキストの一部（？）

- cleaned_label

dataset_labelをクリーニングしたもの

## Discussion

| 題名  | 内容  |
| :---: | :---: |
|  [A List Of Kaggle NLP Competitions & solutions](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/228227)     |   過去のNLPコンペで使われた解放集。Google　Quest　Q＆Aコンペが似たようなタスクだと思う。    |
|[All you wanted to know about key phrase extraction and NER and you were too afraid to ask](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/228376)|NER(named entity recognition)に関するブログや論文へのリンク。いかにしてキーワードとなるような単語を抽出するかが書かれている。|
|[More train data issues](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/discussion/228337)|データセット名とクリーニングされたラベル名の対応に対する質問．複数のクリーニングされたラベルが1つのデータセット名に対応してるじゃんって言われてる|

## Notebooks

|                                                                      題名                                                                      |        内容         |
| :--------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------: |
| [Coleridge Initiative - EDA + Naïve Submission 📚](https://www.kaggle.com/josephassaker/coleridge-initiative-eda-na-ve-submission?select=train) | EDAのお手本にする。 |

## Links

| 題名  | 内容  |
| :---: | :---: |
| [100 Must-Read NLP Papers](https://github.com/mhagiwara/100-nlp-papers)|自然言語処理の論文で重要なものを100本集めたもの。とりあえず基礎知識をつけるために読んで行きたいと思った。       |

## Logs

### 3/28

NotebooksのColeridge Initiative - EDA + Naïve Submission 📚を参考（まるパクリ）にして色々データを触ってみた。

WordCloudで単語の出現頻度を可視化できたのは面白かったけど、小さくまとめられるのがちょっと...って感じだった。

一応、ここから読み取れることが多そうなので画像として保存してから、眺めてみてもいいかもしれない。

### 4/14

ベースラインモデルを作った。アプローチは論文の文章の中に、データセットの名前があるかどうかを調べる簡単なもの。

とりあえず、形態素解析してからもう一度作り直そうと思う。

# 4/19

色々調べてた。モデルの作り方がいまいちわからない...

# 4/25

シンプルな方法でも上位に食い込めることがわかった。

stopwordsを導入してもう一度前処理を直していきたい

dataset_labelsも確認してからsubmissionファイルを作る

# 5/5

NERのブログに目を通してみたけど、これがこのコンペにどのように使えばいいのか良いイメージがわかない。

各論文のタイトル毎にNERで語を抽出して、その語が含まれているかどうかを調べてみる方針もありかもしれないと思った。
