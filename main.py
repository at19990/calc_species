import pandas as pd
import numpy as np
import numpy.random.common
import math
import matplotlib.pyplot as plt
from sklearn import manifold
import random
import datetime
import sys
import os
import numpy.random.bounded_integers
import numpy.random.entropy
import sklearn.neighbors.typedefs
import sklearn.utils._cython_blas
import sklearn.neighbors.quad_tree
import sklearn.tree._utils
import japanize_matplotlib


# U のうちの、総和要素を計算
def sum(f_plus_one, f_plus_one_list, f_plus_two, f_plus_two_list, com_sp, com_sp_list, num_i, num_j, i, j):
    sum_1 = 0
    sum_2 = 0

    for k in com_sp_list:
        sum_1 += (df.iloc[k, i] / num_i)
        # 比較対象調査地では 1 個体のとき
        if df.iloc[k, j] == 1:
            sum_2 += (df.iloc[k, i] / num_i)

    return sum_1, sum_2

# U を計算
def calc_u(f_plus_one, f_plus_one_list, f_plus_two, f_plus_two_list, com_sp, com_sp_list, num_i, num_j, i, j):
    sum_1, sum_2 = sum(f_plus_one, f_plus_one_list, f_plus_two, f_plus_two_list, com_sp, com_sp_list, num_i, num_j, i, j)
    U[i, j] = sum_1 + ((num_j - 1)/ num_j) * (f_plus_one / 2 * f_plus_two) * sum_2

# 選択した列のパラメータを計算
def calc_param(i, j):
    # 比較対象の調査地で 1 個体しか見つからなかった種の数とインデックス
    f_plus_one = 0
    f_plus_one_list = []
    # 比較対象の調査地で 1 個体しか見つからなかった種の数とインデックス
    f_plus_two = 0
    f_plus_two_list = []

    # 各調査地の総個体数
    num_i = df.iloc[:, i].sum(axis=0)
    num_j = df.iloc[:, j].sum(axis=0)

    # 共通種数のそのインデックス
    com_sp = 0;
    com_sp_list = []

    # それぞれカウントする
    for k in range(0, len(df)):
        if df.iloc[k, i] > 0 and df.iloc[k, j] > 0:
            com_sp += 1
            com_sp_list.append(k)

            if df.iloc[k, i] > 0 and df.iloc[k, j] == 1:
                f_plus_one += 1
                f_plus_one_list.append(k)
            if df.iloc[k, i] > 0 and df.iloc[k, j] == 2:
                f_plus_two += 1
                f_plus_two_list.append(k)

    calc_u(f_plus_one, f_plus_one_list, f_plus_two, f_plus_two_list, com_sp, com_sp_list, num_i, num_j, i, j)

# Chao 指数の計算
def calc_chao(i, j):
    # 逆行列
    inv_U = U.T

    # NaN 回避
    if U[i, j] == 0 and inv_U[i, j] == 0:
        J[i, j] = 1
    else:
        J[i, j] = 1 - (U[i, j] * inv_U[i, j]) / (U[i, j] + inv_U[i, j] - (U[i, j] * inv_U[i, j]))


filename = sys.argv[1]

# output フォルダが存在しなければ作成する
if not os.path.exists('output'):
    os.mkdir('output')

# ファイル名と拡張子以降を分割
name, ext = os.path.splitext(filename)

# 現在の日時を取得してフォーマットを変換
dt_now = datetime.datetime.now()
dt_now = str(dt_now.strftime('%Y%m%d-%H%M%S'))

# csv を読み込み、調査地の数・名称を取得する
df = pd.read_csv(filename, index_col=0, encoding="utf8")
sites = len(df.columns)
sites_index = list(df.columns)

# 出力時に用いるインデックスを用意
div_index = ["総個体数", "共通種数", "Shimpson lambda", "Shannon-Wiener H'"]

# 配列を0埋めで初期化
U = np.zeros((sites, sites), dtype=np.float)
J = np.zeros((sites, sites), dtype=np.float)

# パラメータを計算
for i in range(0, sites):
    for j in range(0, sites):
        if i == j:
            pass
        else:
            calc_param(i, j)

# Chao指数を計算
for i in range(0, sites-1):
    for j in range(i + 1, sites):
        if i == j:
            pass
        else:
            calc_chao(i, j)

# Chao指数の計算結果を丸め、 DataFrame に書き出し
df2 = pd.DataFrame(np.round(J, decimals=2), columns=sites_index, index=sites_index)
# データのない不要な列をドロップする
df2 = df2.drop(columns=df2.columns[0])
df2 = df2.drop(index=df2.index[len(df2) - 1])
# 逆行列に変換し、 csv に書き出し (utf-8 であることを明示)
df2 = df2.T
df2.to_csv("./output/{0}_sim_{1}.csv".format(name, dt_now), encoding="utf_8_sig")

# 多様度を入れる配列を0埋めで初期化
div_s = np.zeros((sites, 1), dtype=np.float)
div_h = np.zeros((sites, 1), dtype=np.float)
# 総種数・共通種数を入れる配列を0埋めで初期化
sp_s = np.zeros((sites, 1), dtype=np.float)
com_s = np.zeros((sites, 1), dtype=np.float)

# 出力画像の調査地名の位置をずらし、重ならないように調整するための乱数を用意
rand = random.sample(range(sites), k=sites)

# lambda を計算
for i in range(0, sites):
    sum = 0
    for j in range (0, len(df)):
        # 列の総個体数
        sum_c =  df.iloc[:, i].sum(axis=0)
        # 1個体も種がいない場合パス
        if df.iloc[j, i] == 0:
            pass
        else:
            sum += (df.iloc[j, i] / sum_c)**2
            sp_s[i] += df.iloc[j, i]
            com_s[i] += 1

    div_s[i] = 1 - sum

# H' を計算
for i in range(0, sites):
    sum = 0
    for j in range (0, len(df)):
        sum_c =  df.iloc[:, i].sum(axis=0)
        if df.iloc[j, i] == 0:
            pass
        else:
            # logの底は 2, 10, e などバリエーションがあるようだが、今回は (2008, 大垣)にならって 2 で計算
            sum += (df.iloc[j, i] / sum_c) * math.log2(df.iloc[j, i] / sum_c)

    div_h[i] = sum * (-1)

# 全ての配列を結合
c = np.concatenate((sp_s, com_s, div_s, div_h), axis = 1)

# 丸めて csv に書き出し
df3 = pd.DataFrame(np.round(c, decimals=2), index=sites_index, columns=div_index).T
df3.to_csv("./output/{0}_div_{1}.csv".format(name, dt_now), encoding="utf_8_sig")

# MDS を適用するために Chao 指数の行列に逆行列を足して対称形に整形
datum = J + J.T

# MDS
mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
pos = mds.fit_transform(datum)
plt.scatter(pos[:, 0], pos[:, 1], marker = 'o')

# 画像に書き出し
for i, (label, x, y) in enumerate(zip(sites_index, pos[:, 0], pos[:, 1])):

    # 偶数なら右下に調査地名を振る
    if i%2 == 0:
         plt.annotate(
             label,
             xy = (x, y), xytext = (15+rand[i]*2, -15-rand[i]),
             textcoords = 'offset points', #ha = 'right', va = 'bottom',
             bbox = dict(boxstyle = 'round,pad=0', fc = 'white', alpha = 0.3),
             arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
         )
    # 奇数なら右上に調査地名を振る
    else:
        plt.annotate(
            label,
            xy = (x, y), xytext = (15+rand[i], 15+rand[i]*2),
            textcoords = 'offset points', #ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0', fc = 'white', alpha = 0.3),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
        )
# グラフのタイトル
plt.title("MDS")
# 画像を保存
plt.savefig("./output/{0}_MDS_{1}.png".format(name, dt_now))
