from multiprocessing import Pool
import time
import pickle
import argparse
import sys
sys.path.append('../')
from util import *
import os
from tqdm import tqdm, trange

os.environ["CUDA_VISIBLE_DEVICES"]="2"

parser=argparse.ArgumentParser(description='topN pairs gen')
parser.add_argument('--input_idx', type=str, help='file')
parser.add_argument('--input_line_emb', type=str, help='file')
parser.add_argument('--Model',type=str, help='line, sbert, weighted-sum')
parser.add_argument('--topN', type=int, help='topN')
parser.add_argument('--p', type=float, help='p*(LINE) + (1-p)*SBert')
parser.add_argument('--output_path', type=str, help='file')
args=parser.parse_args()



# Filter pairs in cosine similarity matrix based on topN
def compute_topN(arr, arr2): # Compute topN
    row_idx = int(arr[-1])
    if row_idx in vod_title_idx:
        row = arr[0:-1]
        arr2 = arr2[0:-1]
        col = [[j, row[j], arr2[j]] for j in tv_title_idx]
        col.sort(key = lambda x: x[1]) # sort from smallest to largest
        if col[-1][0] == row_idx: # Do not count (row_idx = col_idx) i.e : exact same between vod and tv
            col = col[:-1]

        for i in col[-topN:]:
            with open(save_path, 'a+') as f:
                if args.Model == 'line':
                    # vod \t tv \t line-score \t sbert-score
                    f.write(f'{row_idx}\t{i[0]}\t{round(i[1], 5)}\t{round(i[2],5)}\n')
                elif args.Model == 'sbert':
                    # vod \t tv \t sbert-score \t line-score
                    f.write(f'{row_idx}\t{i[0]}\t{round(i[1], 5)}\t{round(i[2],5)}\n')
                elif args.Model == 'weighted-sum' or args.Model == 'fusion':
                    # vod \t tv \t weighted-sum-score \t line-score
                    f.write(f'{row_idx}\t{i[0]}\t{round(i[1], 5)}\t{round(i[2],5)}\n')

start_time = time.time()
# Read title index map
i2e, e2i = index_map(args.input_idx) 

# Read LINE embedding
Embedding_line = read_emb(args.input_line_emb, len(i2e))

# Read SBert embedding
model = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens")
title = [i2e[j] for j in range(len(i2e))]
Embedding_sbert = model.encode(title)

print(f'load SBert embedding done: {Embedding_sbert.shape}')

# Top-N
topN = args.topN

# Output txt path
save_path = args.output_path

# read tv title index
tv_title_idx = get_title('../../dataset/v4_kg/tv_triplet.txt', e2i)
# read vod title index
vod_title_idx = get_title('../../dataset/v4_kg/vod_triplet.txt', e2i)

# Remove the existed file 
if os.path.exists(save_path):
        os.remove(save_path)

# Calculate cosine similarity matrix 
print("Calculate cosine similarity matrix... ")

batch_size = 1000

Score_matrix_line = np.zeros((batch_size, len(Embedding_line)), dtype=float)
Score_matrix_sbert = np.zeros((batch_size, len(Embedding_line)), dtype=float)

chunks = 60
pool = Pool(processes=chunks)

for ent_id in trange(0, len(Embedding_line), batch_size):

    if (ent_id + batch_size) > len(Embedding_line):
        batch_size = len(Embedding_line) - ent_id

    Score_matrix_line = calculate_similarity_matrix_batch(ent_id, batch_size, Embedding_line)
    Score_matrix_sbert = calculate_similarity_matrix_batch(ent_id, batch_size, Embedding_sbert)

    # Fill index of each row in new appended last column
    if args.Model == 'weighted-sum':
        p = 0.5
        Score_weighted_sum = p*Score_matrix_line + (1-p)*Score_matrix_sbert  # Compute p*line + (1-p)*sbert
        Score_weighted_sum = np.concatenate((Score_weighted_sum, np.array([[i for i in range(ent_id, ent_id + batch_size)]]).T), axis=1)

    else:
        Score_matrix_line = np.concatenate((Score_matrix_line, np.array([[i for i in range(ent_id, ent_id + batch_size)]]).T), axis=1)
        Score_matrix_sbert = np.concatenate((Score_matrix_sbert, np.array([[i for i in range(ent_id, ent_id + batch_size)]]).T), axis=1)

    if args.Model == 'line':    
        pool.starmap(compute_topN, zip(Score_matrix_line, Score_matrix_sbert))    
    if args.Model == 'sbert':
        pool.starmap(compute_topN, zip(Score_matrix_sbert, Score_matrix_line))   
    if args.Model == 'weighted-sum':
        pool.starmap(compute_topN, zip(Score_weighted_sum, Score_matrix_line))
    # pool.close()

pool.close()
print("--- %s seconds ---" % (time.time() - start_time))