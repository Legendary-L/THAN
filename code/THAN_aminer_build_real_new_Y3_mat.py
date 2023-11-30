import re

import numpy as np
import scipy.io
APA = np.full((22942,22942),0)
# PYP = np.full((18181,18181),0)
APCPA = np.full((22942,22942),0)
DataMat = []
het_walk_f = open("../data/aminer/oriData/" + "THAN_aminer_Y3.txt", "r")
# x=0

for line in het_walk_f:
    curLine = line.strip().split(" ")
    DataMat.append(curLine)
    print(curLine)
    line_id = 0
    for i in DataMat[0]:
        cur=str(DataMat[0][line_id])
        temp = line_id
        # print(line_id)
        line_id = line_id + 1

        if cur[0] == 'p':
            continue
        elif cur[0] == 'a':
            if temp+1 == 512:
                break
            else:
                cur_next=str(DataMat[0][temp+1])
                cur_next_mext=str(DataMat[0][temp+2])
                if cur_next[0] == 'p' and cur_next_mext[0] == 'a':
                     pre=(str(DataMat[0][temp]))[1:]
                     later=(str(DataMat[0][temp+2]))[1:]
                     if int(pre)>22941 or int(later)>22941:
                         continue
                     else:
                         APA[int(pre)][int(later)]=1
                else:
                    continue
        elif cur[0] == 'c':
            if temp+1 == 512:
                break
            else:
                cur_pre = str(DataMat[0][temp-1])
                cur_pre_pre = str(DataMat[0][temp-2])
                cur_next = str(DataMat[0][temp + 1])
                cur_next_mext = str(DataMat[0][temp + 2])
                if cur_pre_pre[0]=='a' and cur_pre[0]=='p' and cur_next[0]=='p' and cur_next_mext[0]=='a':
                    pre = (str(DataMat[0][temp - 2]))[1:]
                    later = (str(DataMat[0][temp + 2]))[1:]
                    if int(pre)>22941 or int(later)>22941:
                        continue
                    else:
                        APCPA[int(pre)][int(later)] = 1
                else:
                    continue
    DataMat=[]

#处理train_idx      val_idx       test_idx
train_idx = np.full((1,18354),0)
val_idx = np.full((1,2294),0)
test_idx = np.full((1,2294),0)

j=0
for i in train_idx[0]:
    if j<6119:
        train_idx[0][j]=j-1
    elif  6119<=j<12237:
        train_idx[0][j]=j+1529
    else:
        train_idx[0][j] = j + 3059
    j=j+1


j=0
for i in val_idx[0]:
    if j<766:
        val_idx[0][j]=j+6117
    elif 766<=j<1531:
        val_idx[0][j]=j+13000
    else:
        val_idx[0][j] = j + 19883
    j=j+1

j=0
for i in test_idx[0]:
    if j<766:
        test_idx[0][j]=j+6882
    elif 766<=j<1531:
        test_idx[0][j]=j+13765
    else:
        test_idx[0][j] = j + 20647
    j=j+1


#处理feature
feature=np.full((22942,1),0)




#处理label

label=np.full((22942,5),0)

het_walk_f = open("../data/aminer/oriData/" + "author_label.txt", "r")
# x=0
# het_walk_f = open("data/acm/" + "productData3.1.txt", "r")

for line in het_walk_f:
    line = line.strip()
    node_id = str(re.split("\t", line)[0])
    neigh_list = str(re.split("\t", line)[1])
    if int(node_id) > 22941:
        continue
    else:
        label[int(node_id)][int(neigh_list)] = 1
# print(x)
# print(PAP[10541][7171])
# print(PCP[1075][10700])
# print(PYP[10557][9776])

# scipy.io.savemat('data2.mat',mdict = {'PAP':PAP,'PCP':PCP,'PYP':PYP,'train_idx':train_idx,'val_idx':val_idx,'test_idx':test_idx,'feature':feature,'label':label})
scipy.io.savemat('aminer3.mat',mdict = {'APA':APA,'APCPA':APCPA,'train_idx':train_idx,'val_idx':val_idx,'test_idx':test_idx,'feature':feature,'label':label})