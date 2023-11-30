import re

import numpy as np
import scipy.io
BSB = np.full((2614,2614),0)
BRURB = np.full((2614,2614),0)
DataMat = []
het_walk_f = open("../data/yelp/" + "X_brurb_1.txt", "r")
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
            if int(pre) > 2614 or int(later) > 2614:
                continue
            else:
                BRURB[int(pre)][int(later)] = 1
                # print(BRURB[int(pre)][int(later)])
    DataMat = []


# for i in range(2614):
#     for j in range(2614):
#         if BRURB[i][j]>5:
#             BRURB[i][j]=1
#         else:
#             BRURB[i][j]=0


het_walk_f = open("../data/yelp/" + "brurb.txt", "r")

for line in het_walk_f:
    # print("能到这吗")
    line = line.strip()
    node_id = str(re.split(" ", line)[0])
    neigh_list = str(re.split(" ", line)[1])
    BRURB[int(node_id)][int(neigh_list)] = 1


het_walk_f = open("../data/yelp/" + "X_bsb_1.txt", "r")
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
            if int(pre) > 2614 or int(later) > 2614:
                continue
            else:
                BSB[int(pre)][int(later)] = 1
                # print(BSB[int(pre)][int(later)])
    DataMat = []
# for i in range(2614):
#     for j in range(2614):
#         if BSB[i][j]>5:
#             BSB[i][j]=1
#         else:
#             BSB[i][j]=0

het_walk_f = open("../data/yelp/" + "bsb.txt", "r")
for line in het_walk_f:
    line = line.strip()
    node_id = str(re.split(" ", line)[0])
    neigh_list = str(re.split(" ", line)[1])
    BSB[int(node_id)][int(neigh_list)] = 1












#处理train_idx      val_idx       test_idx
train_idx = np.full((1,2092),0)
val_idx = np.full((1,261),0)
test_idx = np.full((1,261),0)

j=0
for i in train_idx[0]:
    if j<698:
        train_idx[0][j]=j-1
    elif  698<=j<1395:
        train_idx[0][j]=j+173
    else:
        train_idx[0][j] = j + 347
    j=j+1


j=0
for i in val_idx[0]:
    if j<88:
        val_idx[0][j]=j+696
    elif 88<=j<175:
        val_idx[0][j]=j+1480
    else:
        val_idx[0][j] = j + 2265
    j=j+1

j=0
for i in test_idx[0]:
    if j<88:
        test_idx[0][j]=j+783
    elif 88<=j<175:
        test_idx[0][j]=j+1567
    else:
        test_idx[0][j] = j + 2352
    j=j+1


#处理feature
feature=np.full((2614,1),0)




#处理label

label=np.full((2614,3),0)

het_walk_f = open("../data/yelp/" + "business_category.txt", "r")
for line in het_walk_f:
    line = line.strip()
    node_id = str(re.split("\t", line)[0])
    neigh_list = str(re.split("\t", line)[1])
    label[int(node_id)][int(neigh_list)] = 1


scipy.io.savemat('yelp1.mat',mdict = {'BRURB':BRURB,'BSB':BSB,'train_idx':train_idx,'val_idx':val_idx,'test_idx':test_idx,'feature':feature,'label':label})

