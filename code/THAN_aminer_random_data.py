import argparse
import re
import random

parser = argparse.ArgumentParser(description='input data process')
parser.add_argument('--A_n', type=int, default=22942,
                    help='number of author node')
parser.add_argument('--P_n', type=int, default=18182,
                    help='number of paper node')
parser.add_argument('--C_n', type=int, default=22,
                    help='number of conf node')
parser.add_argument('--Y_n', type=int, default=16,
                    help='number of year node')
parser.add_argument('--data_path', type=str, default='../data/aminer/oriData/',
                    help='path to data')
parser.add_argument('--walk_n', type=int, default=30,
                    help='number of walk per root node')
parser.add_argument('--walk_L', type=int, default=800,
                    help='length of each walk')

args = parser.parse_args()
print(args)


class input_data(object):
    def __init__(self, args):
        self.args = args
        p_neigh_list_train = [[] for k in range(args.P_n)]
        author_paper = [[] for k in range(args.A_n)]
        conf_paper = [[] for k in range(args.C_n)]
        year_paper = [[] for k in range(args.Y_n)]

        # 处理a p
        het_walk_f = open("../data/aminer/oriData/" + "paper_author.txt", "r")

        for line in het_walk_f:
            line = line.strip()
            node_id = str(re.split("\t", line)[0])
            neigh_list = str(re.split("\t", line)[1])
            author_paper[int(neigh_list)].append('p' + node_id)

        # 处理c p

        het_walk_f = open("../data/aminer/oriData/" + "paper_conf.txt", "r")

        for line in het_walk_f:
            line = line.strip()
            node_id = str(re.split("\t", line)[0])
            neigh_list = str(re.split("\t", line)[1])
            conf_paper[int(neigh_list)].append('p' + node_id)

        # 处理y p
        het_walk_f = open("../data/aminer/oriData/" + "paper_year.txt", "r")

        for line in het_walk_f:
            line = line.strip()
            node_id = str(re.split("\t", line)[0])
            neigh_list = str(re.split("\t", line)[1])
            year_paper[int(neigh_list)].append('p' + node_id)

        # 处理p a c y
        relation_f = ["paper_author.txt", "paper_conf.txt"]
        # for i in range(args.P_n):
        # 	p_neigh_list_train[i].append('p' + str(i))

        pre = '0'
        for i in range(len(relation_f)):
            f_name = relation_f[i]
            neigh_f = open(self.args.data_path + f_name, "r")
            for line in neigh_f:
                line = line.strip()
                node_id = re.split("\t", line)[0]
                neigh_list = re.split("\t", line)[1]
                if f_name == 'paper_author.txt':
                    if node_id == pre:
                        p_neigh_list_train[int(pre)].append('a' + neigh_list)
                    else:
                        p_neigh_list_train[int(node_id)].append('a' + neigh_list)
                    pre = node_id
                elif f_name == 'paper_conf.txt':
                    p_neigh_list_train[int(node_id)].append('c' + neigh_list)

            neigh_f.close()
        print(p_neigh_list_train[5])
        print(author_paper[3])
        print(conf_paper[2])
        print(year_paper[1])
        self.p_neigh_list_train = p_neigh_list_train
        self.author_paper = author_paper
        self.conf_paper = conf_paper
        self.year_paper = year_paper

    def gen_het_rand_walk(self):
        print("到这了？？")
        het_walk_f = open(self.args.data_path + "THAN_aminer_random_data.txt", "w")

        for i in range(self.args.walk_n):
            print("到了第几个循环：", end=" ")
            print(i)

            curNode = "N" + str(i)
            het_walk_f.write(curNode + "\n")

            # q = 1
            Q = self.args.walk_L
            # for j in range(self.args.Y_n):
                # print("到了哪个年份： ", end=" ")
                # print(j)

                # if len(self.year_paper[j]):
            for x in range(len(self.year_paper)):
                curNode = "y" + str(x)
                het_walk_f.write(curNode + "\n")
                print("到了哪个年份： ",end=" ")
                print(curNode)

                for y in range(len(self.year_paper[x])):
                    curNode = self.year_paper[x][y]
                    het_walk_f.write(curNode + " ")
                    for l in range(Q - 1):
                        if curNode[0] == "y":
                            curNode = int(curNode[1:])
                            curNode = random.choice(self.year_paper[curNode])
                            het_walk_f.write(curNode + " ")
                        # print(curNode)
                        elif curNode[0] == "a":
                            curNode = int(curNode[1:])
                            curNode = random.choice(self.author_paper[curNode])
                            het_walk_f.write(curNode + " ")
                        # print(curNode)
                        elif curNode[0] == "c":
                            curNode = int(curNode[1:])
                            curNode = random.choice(self.conf_paper[curNode])
                            het_walk_f.write(curNode + " ")
                        # print(curNode)
                        elif curNode[0] == "p":
                            curNode = int(curNode[1:])
                            curNode = random.choice(self.p_neigh_list_train[curNode])
                            het_walk_f.write(curNode + " ")
                        # print(curNode)

                    het_walk_f.write("\n")
                Q = int(Q * 0.8)
            het_walk_f.write("\n")

        het_walk_f.close()


input_data_class = input_data(args=args)

input_data_class.gen_het_rand_walk()
