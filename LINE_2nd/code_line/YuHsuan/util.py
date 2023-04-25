import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from PIL import Image, ImageDraw
from numpy.linalg import norm
from sklearn.metrics import ndcg_score, dcg_score

class CustomError(Exception):
    pass

# def cosine_similarity(A, B):
#     cosine = np.dot(A,B)/(norm(A)*norm(B))
#     return cosine

def read_txt(path):
    f = open(path,'r')
    lines = f.readlines()
    L = []
    for line in lines:
        l = line.strip('\n').split('\t')
        L += [l]
    L = np.array(L)
    return L

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

# Read unified kg
def read_kg(path):
    f = open(path,'r')
    lines = f.readlines()
    L = []
    for line in lines:
        l = line.strip('\n').split(' ')
        L += [l]
    L = np.array(L)
    return L[:,0:2]

# read emb
def read_emb(path, length):
    f = open(path, 'r') 
    lines = f.readlines()
    dim = len(lines[1].replace("\t", " ").strip('\n').split(' ')[1:])
    T = np.zeros((length, dim), dtype=float)
    for line in lines[1:]:
        l = line.replace("\t", " ").strip('\n').split(' ')
        id_ = l[0]
        if id_[0]=='e':
            T[int(id_[1:])] = np.array(l[1:], dtype=float)
    print('(# Title, dim) = ',T.shape)
    return T

# PR
def precision_recall_thres(pair_fil, kk_ans, threshold):
    L = round(len(pair_fil) * threshold)
    pair_fil = pair_fil[0:L+1]
    rule_ans_pair = set([tuple(i) for i in pair_fil]).intersection(kk_ans) 
    num_pair = len(rule_ans_pair)
    if num_pair == 0:
        re, pre, f1 = 0.0, 0.0, 0.0
    else:
        re, pre = num_pair/len(kk_ans), num_pair/len(pair_fil)
        f1 = round(2*(pre*re)/(pre+re) , 4)
        re, pre = round(re,4), round(pre,4)
    return [re, pre, f1]


# 光譜
def create_color_barcode(colors, bar_width, height, width, fname):
    barcode_width = len(colors) * bar_width
    bc = Image.new('RGB', (barcode_width, height))
    draw = ImageDraw.Draw(bc)

    # draw the new barcode
    posx = 0
    # print('Generating barcode...')
    for color in colors:
        draw.rectangle([posx, 0, posx + bar_width, height], fill=color)
        posx += bar_width

    del draw

    bc = bc.resize((width, height), Image.ANTIALIAS)
    bc.save(f'{fname}.PNG')
    return bc

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

# calculate cosine similarity matrix
def similarity_by_chunk(start, end, matrix_len, vector):
    if end > matrix_len:
        end = matrix_len
    return cosine_similarity(X=vector[start:end], Y=vector) 

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