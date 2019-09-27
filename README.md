# calc_species  

## About  
群集組成を記述した地点と種のcsvファイルから各地点の生物種の多様度(Shimpson lambda および Shannon-Wiener H')・各地点の類似度(Chao指数)を計算し、CSVで出力します  
また、類似度からMDS(多次元尺度法)で群集同士の関連性を導いて2次元平面上に図示し、画像で出力します  
実行結果のサンプルは `output` フォルダをごらんください (`sample_utf8.csv` を入力した際の出力結果)

## 使用方法
`sample_utf8.csv` を参考に、列に地点・行に種をとって記述した群衆組成のデータをCSV(エンコード: UTF-8)で保存し、コマンドで

```
python main.py ***.csv
```

のように実行すると、`output` ディレクトリ以下に分析結果が出力されます (フォルダがないときには自動で作成します)

## 出力結果

```
***_div_{実行日時}.csv : 各地点における多様度

***_sim_{実行日時}.csv : 各地点間の類似度

***_MDS_{実行日時}.png : 多次元尺度法による分析結果
```

## 参考文献

[[1]](https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1461-0248.2004.00707.x) Chao A, Chazdon RL, Colwell RK, Shen T (2005) A new statistical approach for assessing similarity of species composition with incidence and abundance data. Ecol Let 8:148-159

[[2]](https://www.jstage.jst.go.jp/article/seitai/61/1/61_KJ00007176266/\_pdf/-char/ja) 土居秀幸, 岡村寛 (2011) 生物群集解析のための類似度とその応用：R を使った類似度の算出、グラフ化、検定. 日本生態学会誌 61：3 - 20

[[3]](http://www.mus-nh.city.osaka.jp/iso/argo/nl15/nl15-10-22.pdf) 大垣俊一 (2008) 多様度と類似度、分類学的新指標. Argonauta 15:10-22

[[4]](http://www.kaiseiken.or.jp/study/lib/news123kaisetu.pdf) 山田 裕 (2014) 生物の多様度指数(1). 海生研ニュース 123 : 6 - 7  

## Note  
リファクタリング・検証はこれから進める予定です  
出力結果の正確性および利用に伴い発生した損害に関して、本プログラムの製作者は責任を負いかねます
