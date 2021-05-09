[Combining Labeled and Unlabeled Data with Co-Training](https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf)

1. どんなもの？

ラベルがついているデータとラベルのついていないデータの両方を扱う方法についての論文．

webテキストの分類において，ラベルなしデータの方が集めるのが簡単であるためどのようにしてラベルなしデータを扱うかと言うのは重要な点である．

PAC手法で学習している．

2. intro(abstは1に書いたままなので)

ウェブページは2つの種類に分けることができる．

    1. ウェブページそのものに現れる単語
    2. そのウェブページのリンクに現れる単語

モデルの学習には，最初にラベル付きデータを使って弱い分類器を作る．

次に，ラベルなしデータに弱い分類器を適用して出力を次の学習の入力にする．

3. conclusion

このモデルで現実の問題を扱うにはシンプル過ぎた．

データの相関はどのくらいまで許されるのか？という疑問は残る

ある程度は，実世界の問題にも応用できる可能性がある．

4. 感想

このコンペには特に関係なさそうだと思った．

英語の意味をうまく掬いきれなかった気がする.



## TAMMIC for content of paper

- Title

[Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)

- Author

John Lafferty, Andrew McCallum, Fernando Pereira

- Motivation

HMMや確率的文法は多くの問題に適用されてきた．

でも，対象が複数の状態や長期の依存関係を持つ場合実用的ではない．

そのため，推論問題において手に負えなくなってしまう問題がある．

- Method

与えられた対象から，全体のラベルの組み合わせの確率を計算する．

ランダムなデータの系列Xと，それに対応したラベル系列Yの条件付き確率P(Y|X)を最大化するように計算する．

P(X)を最大化する訳ではない．

- Insight

AdaBoostに使われるexponential loss functionを用いた．少ないラベル数の分類問題に用いられるものだが，系列あたりの損失を最適化することに適用できた．（順伝播，逆伝播ネットワークが必要）

効率的に素性選択やfeature induction(素性導入？)のアルゴリズムが作成できる．

- Contribution Summary

## KURR for next paper

- Keyword

    CRF, MEMM, HMM, 確率的文法

- Unknown

    他の学習の方法を探る

- Reflection

    単語の品詞タグ付けに活用されているから，文章全体を読み込ませて論文のラベル付けができるかな？

- Reference

    わからん
