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




def sum(f_plus_one, f_plus_one_list, f_plus_two, f_plus_two_list, com_sp, com_sp_list, num_i, num_j, i, j):
    sum_1 = 0
    sum_2 = 0

    for k in com_sp_list:
        sum_1 += (df.iloc[k, i] / num_i)

        # print("df.iloc[{0}, {1}] = {2}".format(k, i, df.iloc[k, i]))

        if df.iloc[k, j] == 1:
            sum_2 += (df.iloc[k, i] / num_i)

    # print("sum_1 = " + str(sum_1))
    # print("sum_2 = " + str(sum_2))

    return sum_1, sum_2


def calc_u(f_plus_one, f_plus_one_list, f_plus_two, f_plus_two_list, com_sp, com_sp_list, num_i, num_j, i, j):
    sum_1, sum_2 = sum(f_plus_one, f_plus_one_list, f_plus_two, f_plus_two_list, com_sp, com_sp_list, num_i, num_j, i, j)

    U[i, j] = sum_1 + ((num_j - 1)/ num_j) * (f_plus_one / 2 * f_plus_two) * sum_2


# 選択した列のパラメータを計算
def calc_param(i, j):
    f_plus_one = 0
    f_plus_one_list = []
    f_plus_two = 0
    f_plus_two_list = []

    num_i = df.iloc[:, i].sum(axis=0)
    num_j = df.iloc[:, j].sum(axis=0)

    # print("num_j = {0}, num_i = {1}".format(num_i, num_j))

    com_sp = 0;
    com_sp_list = []


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

    # print("com_sp = {0}, f_plus_one = {1}, f_plus_two = {2}".format(com_sp, f_plus_one, f_plus_two))

    calc_u(f_plus_one, f_plus_one_list, f_plus_two, f_plus_two_list, com_sp, com_sp_list, num_i, num_j, i, j)


def calc_chao(i, j):
    inv_U = U.T

    if U[i, j] == 0 and inv_U[i, j] == 0:
        J[i, j] = 1
    else:
        J[i, j] = 1 - (U[i, j] * inv_U[i, j]) / (U[i, j] + inv_U[i, j] - (U[i, j] * inv_U[i, j]))



filename = sys.argv[1]

if not os.path.exists('output'):
    os.mkdir('output')


name, ext = os.path.splitext(filename)


# filename, ext = os.path.splitext(os.path.basename(filename) )

dt_now = datetime.datetime.now()

dt_now = str(dt_now.strftime('%Y%m%d-%H%M%S'))

#dt_now = str(dt_now.year) + str(dt_now.month) + str(dt_now.day) + "-" + str(dt_now.hour) + str(dt_now.minute) + str(dt_now.second)

df = pd.read_csv(filename, index_col=0, encoding="utf8")

# print(df)

# print(df.dtypes)

sites = len(df.columns)

sites_index = list(df.columns)



div_index = ["Shimpson lambda", "Shannon-Wiener H'"]

# print(sites_index)

"""
print("sites = " + str(sites))
"""

U = np.zeros((sites, sites), dtype=np.float)

J = np.zeros((sites, sites), dtype=np.float)


for i in range(0, sites):
    # print("i = " + str(i))
    for j in range(0, sites):
        if i == j:
            pass
        else:
            # print("i = {0}, j = {1}".format(i, j))
            calc_param(i, j)


# print(U)
# print(U.T)

for i in range(0, sites-1):

    for j in range(i + 1, sites):
        if i == j:
            pass
        else:
            # print("i = {0}, j = {1}".format(i, j))
            calc_chao(i, j)




# df2 = pd.DataFrame(np.round(J, decimals=1))

df2 = pd.DataFrame(np.round(J, decimals=2), columns=sites_index, index=sites_index)

df2 = df2.drop(columns=df2.columns[0])

df2 = df2.drop(index=df2.index[len(df2) - 1])

df2 = df2.T

# print(df2)

df2.to_csv("./output/{0}_sim_{1}.csv".format(name, dt_now), encoding="utf8")

div_s = np.zeros((sites, 1), dtype=np.float)

div_h = np.zeros((sites, 1), dtype=np.float)

rand = random.sample(range(sites), k=sites)

for i in range(0, sites):
    sum = 0
    for j in range (0, len(df)):

        sum_c =  df.iloc[:, i].sum(axis=0)
        # print(sum_c)

        if df.iloc[j, i] == 0:
            pass
        else:
            sum += (df.iloc[j, i] / sum_c)**2
        # print("sum = {0}".format(sum))

    div_s[i] = 1 - sum


for i in range(0, sites):
    sum = 0
    for j in range (0, len(df)):

        sum_c =  df.iloc[:, i].sum(axis=0)


        if df.iloc[j, i] == 0:
            pass
        else:
            sum += (df.iloc[j, i] / sum_c) * math.log2(df.iloc[j, i] / sum_c)
        # print("sum = {0}".format(sum))


    div_h[i] = sum * (-1)

c = np.concatenate((div_s,div_h), axis = 1)

# print(c)

# print(div_s)
# print(div_h)

df3 = pd.DataFrame(np.round(c, decimals=2), index=sites_index, columns=div_index).T
#df3.append(np.round(div_h, decimals=1))
df3.to_csv("./output/{0}_div_{1}.csv".format(name, dt_now), encoding="utf8")

# print(df3)
# print(div_h)

# datum = np.loadtxt("site_sim.csv",delimiter=",",usecols=range(1,len(df.columns)), skiprows=1)
datum = J + J.T

# print(datum)
mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)

pos = mds.fit_transform(datum)

plt.scatter(pos[:, 0], pos[:, 1], marker = 'o')

for i, (label, x, y) in enumerate(zip(sites_index, pos[:, 0], pos[:, 1])):

    if i%2 == 0:
         plt.annotate(
             label,
             xy = (x, y), xytext = (15+rand[i]*2, -15-rand[i]),
             textcoords = 'offset points', #ha = 'right', va = 'bottom',
             bbox = dict(boxstyle = 'round,pad=0', fc = 'white', alpha = 0.3),
             arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
         )
    else:
        plt.annotate(
            label,
            xy = (x, y), xytext = (15+rand[i], 15+rand[i]*2),
            textcoords = 'offset points', #ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0', fc = 'white', alpha = 0.3),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
        )


plt.title("MDS")


plt.savefig("./output/{0}_MDS_{1}.png".format(name, dt_now))
