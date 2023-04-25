import numpy as np
from sklearn.metrics import ndcg_score, dcg_score
import argparse
from multiprocessing import Pool
import pickle
from util import *

def read_txt(path):
    f = open(path,'r')
    lines = f.readlines()
    L = []
    for line in lines:
        l = line.strip('\n').split('\t')
        L += [l]
    L = np.array(L)
    return L

# ground truth to index
def gt2idx(gt):
    gt_idx = []
    gt = list(gt)
    for idx in range(len(gt)):
        e1, e2 = e2i[gt[idx][0]], e2i[gt[idx][1]]
        #if e1 >= e2:
        #    gt_idx.append(tuple([str(e2), str(e1)]))
        #else:
        gt_idx.append(tuple([str(e1), str(e2)]))
    return gt_idx

def filter_gt(p):
    if tuple([p[0], p[1]]) in gt_all:
        return list(p)
    else:
        return None

def ndcg_cal(sorted_pairs, k):
    y_predicted = []
    if k!='all':
        sorted_pairs = sorted_pairs[0:k]
        y_ideal = [2 for i in range(k)]
    for i in sorted_pairs:
        if tuple(i) in gt_same_idx:
            y_predicted.append(2)
        elif tuple(i) in gt_similar_idx:
            y_predicted.append(1)
        elif tuple(i) in gt_notsame_idx:
            y_predicted.append(0)
    if k=='all':
        y_ideal = sorted(y_predicted, reverse = True)

    dcg_ = round(dcg_score(np.asarray([y_predicted]), np.asarray([y_ideal])),4)
    idcg_ = round(dcg_score(np.asarray([y_ideal]), np.asarray([y_ideal])),4)
    ndcg_ = dcg_ / idcg_

    print(f'DCG@{k}: {dcg_}, IDCG@{k}: {idcg_}, NDCG@{k}: {ndcg_} ')
    print()

parser=argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--input_idx', type=str, help='file')
parser.add_argument('--input_pairs', type=str, help='file')
parser.add_argument('--filter', type=int, default=1, help='0: Not filter in GT, 1: filter in GT')
args=parser.parse_args()

# Read title index map
i2e, e2i = index_map(args.input_idx) 

# read result pairs
result_pair = read_txt(args.input_pairs)
print('pairs before filter: ',len(result_pair))

# read ground truth
with open ('../../dataset/labelled_gt/kk_same.pkl', 'rb') as fp:
    gt_same = pickle.load(fp)
with open ('../../dataset/labelled_gt/kk_similar.pkl', 'rb') as fp:
    gt_similar = pickle.load(fp)
with open ('../../dataset/labelled_gt/kk_notsame.pkl', 'rb') as fp:
    gt_notsame = pickle.load(fp)

# ground truth index
gt_same_idx = gt2idx(gt_same)
gt_similar_idx = gt2idx(gt_similar)
gt_notsame_idx = gt2idx(gt_notsame)
gt_all = gt_same_idx + gt_similar_idx + gt_notsame_idx

if args.filter:
    # pool and chunks
    chunks = 60
    pool = Pool(processes=chunks)

    # filter 
    result_pair = pool.map(filter_gt, result_pair)
    result_pair = np.array([i for i in result_pair if i!=None])
    print('pairs after filter: ',len(result_pair))

    # sort from highest score to lowest score
    result_pair = result_pair[result_pair[:, 2].argsort()][::-1]

    # NDCG
    sorted_pairs = result_pair[:,0:2]
    ndcg_cal(sorted_pairs, 'all')
    ndcg_cal(sorted_pairs, 100)
    ndcg_cal(sorted_pairs, 1000)

    # 光譜圖
    colors = []
    for p in sorted_pairs:
        color = "brown"
        if tuple(p) in gt_same_idx:
            color = "green"
        elif tuple(p) in gt_similar_idx:
            color = "royalblue"
        colors.append(color)
    bar_width = 10
    height = 250
    width = 500
    create_color_barcode(colors, bar_width, height, width, "result/spectrum-plot")
else:
    print('Do not filter pairs')
    print('Cannot obtain NDCG')


# PR
rate = [round(0.05 + 0.05*i,2) for i in range(20) ] # calculate precision and recall @ 0.95, 0.9, 0.85, ....
record = []
print('PR in same label')
for i in rate:
    record.append(precision_recall_thres([tuple([i[0],i[1]]) for i in result_pair], gt_same_idx, i))
