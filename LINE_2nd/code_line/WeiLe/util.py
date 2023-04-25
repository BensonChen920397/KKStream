from transformers import BertJapaneseTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from PIL import Image, ImageDraw


# Load SBert
class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)


# read title index map
def index_map(input_path):
    i2e, e2i = {}, {}
    f = open(input_path, 'r')  
    lines = f.readlines()
    for line in lines:
        l = line.strip('\n').split('\t')
        if l[1][0]=='e':
            e2i[l[0]] = int(l[1][1:])
            i2e[int(l[1][1:])] = l[0]
    return i2e, e2i

# read emb
def read_emb(path, length):
    f = open(path, 'r') 
    lines = f.readlines()
    dim = len(lines[1].strip('\n').split(' ')[1:])
    T = np.zeros((length, dim), dtype=float)
    for line in lines[1:]:
        l = line.strip('\n').split(' ')
        id_ = l[0]
        if id_[0]=='e':
            T[int(id_[1:])] = np.array(l[1:], dtype=float)
    print('(# Title, dim) = ',T.shape)
    return T

# check and make file
def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

# For each row in a matrix, calculate the rank for all columns w.r.t this row
def rank(x, axis=-1):
    res = np.empty(x.shape, dtype=int)
    I = np.ogrid[tuple(map(slice, x.shape))]
    rng, I[axis] = I[axis], x.argsort(axis=axis)
    res[I] = rng
    return res

# calculate cosine similarity matrix
def similarity_by_chunk(start, end, matrix_len, vector):
    if end > matrix_len:
        end = matrix_len
    return cosine_similarity(X=vector[start:end], Y=vector) 

# calculate cosine similarity rank matrix
def similarity_rank_by_chunk(start, end, matrix_len, vector):
    if end > matrix_len:
        end = matrix_len
    cos_sim = cosine_similarity(X=vector[start:end], Y=vector) 
    return matrix_len - rank(cos_sim, axis=-1).astype(int) # return rank

def calculate_similarity_matrix(vector):
    chunk_size = 500  # Change chunk_size to control resource consumption and speed, Higher chunk_size means more memory/RAM needed but also faster
    matrix_len = vector.shape[0] 
    Score_mat = np.zeros((matrix_len,matrix_len), dtype=float) # score matrix

    for chunk_start in range(0, matrix_len, chunk_size):
        cosine_similarity_chunk = similarity_by_chunk(chunk_start, chunk_start+chunk_size, matrix_len, vector)
        Score_mat[chunk_start:chunk_start+chunk_size,:] = cosine_similarity_chunk
    return Score_mat

def calculate_similarity_matrix_batch(ent_id, batch_size, vector):
    chunk_size = 500  # Change chunk_size to control resource consumption and speed, Higher chunk_size means more memory/RAM needed but also faster
    matrix_len = vector.shape[0]
    Score_mat = np.zeros((batch_size, matrix_len), dtype=float) # score matrix

    for chunk_start in range(0, batch_size, chunk_size):
        cosine_similarity_chunk = similarity_by_chunk(ent_id + chunk_start, ent_id + chunk_start + chunk_size, matrix_len, vector)
        if(chunk_start+chunk_size > matrix_len):
            Score_mat[chunk_start:matrix_len,:] = cosine_similarity_chunk
        else:
            Score_mat[chunk_start:chunk_start+chunk_size,:] = cosine_similarity_chunk
    return Score_mat

# calculate cosine similarity rank matrix
def calculate_similarity_rank_matrix(vector):
    chunk_size = 500  # Change chunk_size to control resource consumption and speed, Higher chunk_size means more memory/RAM needed but also faster
    matrix_len = vector.shape[0] 
    Score_mat = np.zeros((matrix_len,matrix_len), dtype=float) # score matrix

    for chunk_start in range(0, matrix_len, chunk_size):
        cosine_similarity_chunk = similarity_rank_by_chunk(chunk_start, chunk_start+chunk_size, matrix_len, vector)
        Score_mat[chunk_start:chunk_start+chunk_size,:] = cosine_similarity_chunk
    return Score_mat

def get_title(triplet_path, idx_map):
    title = []
    f = open(triplet_path, 'r')  
    lines = f.readlines()
    for line in lines[1:]:
        l = line.strip('\n').split('\t')
        title.append(l[0])
    title = list(set(title))
    title_idx = [idx_map[j] for j in title]
    return title_idx

# PR
def precision_recall_thres(pair_fil, kk_ans, threshold):
    L = round(len(pair_fil) * threshold)
    pair_fil = pair_fil[0:L+1]
    rule_ans_pair = set([tuple(i) for i in pair_fil]).intersection(kk_ans) 
    num_pair = len(rule_ans_pair)
    re, pre = num_pair/len(kk_ans), num_pair/len(pair_fil)
    f1 = round(2*(pre*re)/(pre+re) , 4)
    re, pre = round(re,4), round(pre,4)
    print(f'recall@{threshold}: {num_pair}/{len(kk_ans)} = {re},    precision@{threshold}: {num_pair}/{len(pair_fil)} = {pre},    f1@{threshold}: {f1}')
    return [re, pre, f1]

# 光譜
def create_color_barcode(colors, bar_width, height, width, fname):
    barcode_width = len(colors) * bar_width
    bc = Image.new('RGB', (barcode_width, height))
    draw = ImageDraw.Draw(bc)

    # draw the new barcode
    posx = 0
    print('Generating barcode...')
    for color in colors:
        draw.rectangle([posx, 0, posx + bar_width, height], fill=color)
        posx += bar_width

    del draw

    bc = bc.resize((width, height), Image.ANTIALIAS)
    bc.save(f'{fname}.PNG')
    return bc