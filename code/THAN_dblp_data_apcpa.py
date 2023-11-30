import six.moves.cPickle as pickle
import numpy as np
import string
import re
import random
import math
from collections import Counter
from itertools import *

walk_n = 30
walk_L = 24     #80/5 80步 5个节点一步
Y_n=3


p_neigh_list_train = [[] for k in range(14475)]
author_paper = [[] for k in range(14475)]
# ../data/academic/
het_walk_f = open("../data/dblp/oriData/" + "APCPA_F_T.txt", "r")

for line in het_walk_f:
    line = line.strip()
    node_id = str(re.split(" ", line)[0])
    neigh_list = str(re.split(" ", line)[1])
    author_paper[int(node_id)].append(neigh_list)
#
# for i in range(2614):
#     print(i, end=" ")
#     print(author_paper[i])

het_walk_f = open("../data/dblp/oriData/" + "dblp_data_apcpa.txt", "w")
for i in range(walk_n):
    print("到了第几个循环：", end=" ")
    print(i)
    curNode = "N" + str(i)
    het_walk_f.write(curNode + "\n")
    Q = walk_L
    for x in range(Y_n):
        curNode = "y" + str(x)
        het_walk_f.write(curNode + "\n")
        for y in range(14475):
            # print(self.year_paper[x][y], end=" ")
            curNode = y
            het_walk_f.write(str(curNode) + " ")
            for l in range(Q - 1):
                curNode=int(curNode)
                curNode = random.choice(author_paper[curNode])
                het_walk_f.write(str(curNode) + " ")
            het_walk_f.write("\n")
        Q = int(Q * 0.8)
    het_walk_f.write("\n")
het_walk_f.close()
