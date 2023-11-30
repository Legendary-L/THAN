import re

import numpy as np
import scipy.io
APA = np.full((14475,14475),0)
APCPA = np.full((14475,14475),0)
DataMat = []
het_walk_f = open("../data/dblp/oriData/" + "X_apcpa_1.txt", "r")
for line in het_walk_f:
    curLine = line.strip().split(" ")
    DataMat.append(curLine)
    line_id = 0
    for i in DataMat[0]:
        cur = str(DataMat[0][line_id])
        line_id = line_id + 1
        if line_id + 1 == 16:
            break
        else:
            pre = (str(DataMat[0][line_id - 1]))[0:]
            later = (str(DataMat[0][line_id]))[0:]
            # print("pre later")
            # print(pre)
            # print(later)
            if int(pre) > 14474 or int(later) > 14474:
                continue
            else:
                APCPA[int(pre)][int(later)] = 1
                # print(BRURB[int(pre)][int(later)])
    DataMat = []


het_walk_f = open("../data/dblp/oriData/" + "APCPA.txt", "r")

for line in het_walk_f:
    # print("能到这吗")
    line = line.strip()
    node_id = str(re.split(" ", line)[0])
    neigh_list = str(re.split(" ", line)[1])
    APCPA[int(node_id)][int(neigh_list)] = 1


het_walk_f = open("../data/dblp/oriData/" + "X_apa_1.txt", "r")
for line in het_walk_f:
    curLine = line.strip().split(" ")
    DataMat.append(curLine)
    line_id = 0
    for i in DataMat[0]:
        cur = str(DataMat[0][line_id])
        line_id = line_id + 1
        if line_id + 1 == 16:
            break
        else:
            pre = (str(DataMat[0][line_id - 1]))[0:]
            later = (str(DataMat[0][line_id]))[0:]
            # print("pre later")
            # print(pre)
            # print(later)
            if int(pre) > 14474 or int(later) > 14474:
                continue
            else:
                APA[int(pre)][int(later)] = 1
                # print(BSB[int(pre)][int(later)])
    DataMat = []
# for i in range(2614):
#     for j in range(2614):
#         if BSB[i][j]>5:
#             BSB[i][j]=1
#         else:
#             BSB[i][j]=0

het_walk_f = open("../data/dblp/oriData/" + "APA.txt", "r")
for line in het_walk_f:
    line = line.strip()
    node_id = str(re.split(" ", line)[0])
    neigh_list = str(re.split(" ", line)[1])
    APA[int(node_id)][int(neigh_list)] = 1





#处理train_idx      val_idx       test_idx
train_idx = np.full((1,11580),0)
val_idx = np.full((1,1448),0)
test_idx = np.full((1,1447),0)

j=0
for i in train_idx[0]:
    if j<3861:
        train_idx[0][j]=j-1
    elif  3861<=j<7721:
        train_idx[0][j]=j+965
    else:
        train_idx[0][j] = j + 1931
    j=j+1


j=0
for i in val_idx[0]:
    if j<484:
        val_idx[0][j]=j+3859
    elif 484<=j<967:
        val_idx[0][j]=j+8202
    else:
        val_idx[0][j] = j + 12545
    j=j+1

j=0
for i in test_idx[0]:
    if j<484:
        test_idx[0][j]=j+4342
    elif 484<=j<967:
        test_idx[0][j]=j+8685
    else:
        test_idx[0][j] = j + 13027
    j=j+1


#处理feature
feature=np.full((14475,4),0)




#处理label

label=np.full((14475,4),0)


het_walk_f = open("../data/dblp/oriData/" + "author_label_new2.txt", "r")
for line in het_walk_f:
    line = line.strip()
    node_id = str(re.split("\t", line)[0])
    neigh_list = str(re.split("\t", line)[1])
    label[int(node_id)][int(neigh_list)] = 1


scipy.io.savemat('dblp1.mat',mdict = {'APCPA':APCPA,'APA':APA,'train_idx':train_idx,'val_idx':val_idx,'test_idx':test_idx,'feature':feature,'label':label})

