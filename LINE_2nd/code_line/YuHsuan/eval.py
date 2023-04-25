import pickle
import argparse
import sys
sys.path.append('../../')
import os
import numpy as np
from util import *
from tqdm import tqdm, trange
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.decomposition import PCA

os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser=argparse.ArgumentParser(description=' ')
parser.add_argument('--input_idx', type=str, help='folder : /tmp2/yhchen/KKStream_project/LINE/LINE_2nd/dataset/line_kg/')
parser.add_argument('--input_line_emb', type=str, help='file : kg-2nd-500.txt')
parser.add_argument('--input_sbert_emb', type=str, help='file : twotower.embed', default=None, required=False)
parser.add_argument('--output_path', type=str, help='result/')
parser.add_argument('--output_describe', help='Description of setting of output result', required=False)
parser.add_argument('--dim_reduction', type=int, help='Dimenson after doing pca', default=768, required=False)

args=parser.parse_args()

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

# NDCG
def ndcg_cal(sorted_pairs, thresholds):
    y_label = []
    y_pred = list(sorted_pairs[:,2])
    sorted_pairs = sorted_pairs[:,0:2]
    for vod, tv in sorted_pairs:
        vod = str(int(vod))
        tv = str(int(tv))
        if (vod, tv) in gt_same_idx:
            y_label.append(2)
        elif (vod, tv) in gt_similar_idx:
            y_label.append(1)
        elif (vod, tv) in gt_all_idx:
            y_label.append(0)
   
    dcg_ = round(dcg_score(np.asarray([y_label], dtype=float), np.asarray([y_pred], dtype=float), k=thresholds),4)
    idcg_ = round(dcg_score(np.asarray([y_label], dtype=float), np.asarray([y_label], dtype=float), k=thresholds),4)
    ndcg_ = dcg_ / idcg_

    return dcg_, idcg_, ndcg_

# Spectrum
def Spectrum(score_gt_array, output):
    colors = []
    for i in score_gt_array:
        color = "brown"
        vod = str(int(i[0]))
        tv = str(int(i[1]))
        if tuple([vod, tv]) in gt_same_idx:
            color = "green"
        elif tuple([vod, tv]) in gt_similar_idx:
            color = "royalblue"
        colors.append(color)
    bar_width = 10
    height = 250
    width = 500
    create_color_barcode(colors, bar_width, height, width, output + 'spectrum-plot')

print('Start a new evaluation !!!')

# Read unified kg
unified_kg = read_kg(args.input_idx + 'line-kg.txt')

# Read title index map
i2e, e2i = index_map(args.input_idx + 'line-kg.idx.txt')

# read kg1 title index
kg1_title_idx = get_title('/tmp2/yhchen/KKStream_project/LINE/LINE_2nd/dataset/v4_kg/vod_triplet.txt', e2i)
# read kg2 title index
kg2_title_idx = get_title('/tmp2/yhchen/KKStream_project/LINE/LINE_2nd/dataset/v4_kg/tv_triplet.txt', e2i)

# Read LINE embedding
Embedding_line = read_emb(args.input_line_emb, len(i2e))

# title : meta dictionary
tit2meta = defaultdict(list)
for i in unified_kg:
    if i[1] not in tit2meta[i[0]]:
        tit2meta[i[0]].append(i[1])

# Read LINE embedding
if args.input_sbert_emb != None:
    Embedding_sbert = read_emb(args.input_sbert_emb, len(i2e))
else:
    model = SentenceTransformer('bert-base-multilingual-cased')
    title = [i2e[j] for j in range(len(i2e))]
    Embedding_sbert = model.encode(title)
print(f'load SBert embedding done: {Embedding_sbert.shape}')

# Dimension reduction
pca = PCA(n_components = args.dim_reduction)
Embedding_sbert = pca.fit_transform(Embedding_sbert)
print("The shape of sbert embedding after pca is ", Embedding_sbert.shape)
    
save_path = args.output_path

if not os.path.isdir(save_path):
    os.makedirs(save_path)

# read ground truth
# RAC
with open ('/tmp2/yhchen/KKStream_project/LINE/LINE_2nd/dataset/labelled_gt/rac_gt/same_dif.pkl', 'rb') as fp:
    gt_same = pickle.load(fp)
with open ('/tmp2/yhchen/KKStream_project/LINE/LINE_2nd/dataset/labelled_gt/rac_gt/similar_dif.pkl', 'rb') as fp:
    gt_similar = pickle.load(fp)
with open ('/tmp2/yhchen/KKStream_project/LINE/LINE_2nd/dataset/labelled_gt/rac_gt/kk_same+similar+notsame_dif.pkl', 'rb') as fp:
    gt_all = pickle.load(fp)

# ground truth index
# RAC
gt_same_idx = gt2idx(gt_same)
gt_similar_idx = gt2idx(gt_similar)
gt_all_idx = gt2idx(gt_all)

thresholds_list = [100, 300, 500, 700, 918, 1000, 2000, 3000, 5000, 10000]

# calculate cosine similarity on gt
score_gt = []

# Make Prediction
for vod, tv in gt_all_idx:
    vod = int(vod)
    tv = int(tv)
    vod_embedding = Embedding_line[vod]
    tv_embedding = Embedding_line[tv]
    cosine = np.dot(vod_embedding,tv_embedding)/(norm(vod_embedding)*norm(tv_embedding))
    score_gt.append([vod, tv, cosine])

# Make Prediction(concat with SBERT embedding)
# for vod, tv in gt_all_idx:
#     vod = int(vod)
#     tv = int(tv)
#     vod_embedding = np.concatenate((Embedding_line[vod], Embedding_sbert[vod]), axis=0)
#     tv_embedding = np.concatenate((Embedding_line[tv], Embedding_sbert[tv]), axis=0)
#     cosine = np.dot(vod_embedding,tv_embedding)/(norm(vod_embedding)*norm(tv_embedding))
#     score_gt.append([vod, tv, cosine])

score_gt = np.array(score_gt)
score_gt = score_gt[score_gt[:, 2].argsort()][::-1]
output_filename = f'{save_path}{args.output_describe}' + '_result.txt' if args.output_describe !=None else  f'{args.save_path}' + 'result.txt'
# output_filename = save_path + args.output_describe + 'result.txt'

with open(output_filename, 'w+') as f:

    # NDCG
    f.write(f'NDCG @ Ground Truth \n')
    for threshold in thresholds_list:
        dcg, idcg, ndcg = ndcg_cal(score_gt, threshold)
        print(f'DCG@{threshold}: {dcg}, IDCG@{threshold}: {idcg}, NDCG@{threshold}: {ndcg}')
        f.write(f'DCG@{threshold}: {dcg}, IDCG@{threshold}: {idcg}, NDCG@{threshold}: {ndcg} \n')
    dcg, idcg, ndcg = ndcg_cal(score_gt, None)
    print(f'DCG@all: {dcg}, IDCG@all: {idcg}, NDCG@all: {ndcg}')
    f.write(f'DCG@all: {dcg}, IDCG@all: {idcg}, NDCG@all: {ndcg} \n \n')

    # Recall
    f.write(f'PR in same label \n')
    for threshold in thresholds_list:
        pr_record_same = precision_recall_thres([tuple([str(int(i[0])),str(int(i[1]))]) for i in score_gt[:threshold]], gt_same_idx, threshold=1)
        print(f'recall@{threshold}:{pr_record_same[0]}\t precision@{threshold}:{pr_record_same[1]}\t f1@{threshold}:{pr_record_same[2]}')
        f.write(f'recall@{threshold}:{pr_record_same[0]}\t precision@{threshold}:{pr_record_same[1]}\t f1@{threshold}:{pr_record_same[2]} \n')
    pr_record_same = precision_recall_thres([tuple([str(int(i[0])),str(int(i[1]))]) for i in score_gt], gt_same_idx, threshold=1)
    print(f'recall@all:{pr_record_same[0]}\t precision@all:{pr_record_same[1]}\t f1@all:{pr_record_same[2]}')
    f.write(f'recall@all:{pr_record_same[0]}\t precision@all:{pr_record_same[1]}\t f1@all:{pr_record_same[2]} \n')

print("Done!!! \n")
