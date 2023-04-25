import csv
import numpy as np
import argparse

parser=argparse.ArgumentParser(description='unified kg generator')
parser.add_argument('--input_kg1', type=str, help='input kg1 txt')
parser.add_argument('--input_kg2', type=str, help='input kg2 txt')
parser.add_argument('--output_kg', type=str, help='output kg txt')
args=parser.parse_args()


def read_kg(path):
    f = open(path, 'r')
    L1 = []
    lines = f.readlines()
    for line in lines:
        l = line.strip('\n').split('\t')
        L1+=[l]
    print(f'len of {path}: ',len(L1))
    return L1

def gen_unified_kg(vod_kg, tv_kg, result_path):
    vod_kg = read_kg(vod_kg)[1:]
    tv_kg = read_kg(tv_kg)[1:]
    kg_add = vod_kg + tv_kg
    unified_kg = list(map(list,set(map(tuple,kg_add))))
    print(f'len of unified kg: ',len(unified_kg))
    
    unified_kg = [['ent', 'rel', 'ent']] + unified_kg 
    with open(result_path, "w", newline="") as f: # save
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(unified_kg)
    
    # statistics
    vod_kg = np.array(vod_kg)
    tv_kg = np.array(tv_kg)
    unified_kg = np.array(unified_kg)
    print('# of titles, # of relations, # of ent')
    print(f'vod: {len(set(vod_kg[:,0]))}, {len(set(vod_kg[:,1]))}, {len(set(vod_kg[:,2]))}')
    print(f'tv:  {len(set(tv_kg[:,0]))}, {len(set(tv_kg[:,1]))}, {len(set(tv_kg[:,2]))}')
    print(f'all: {len(set(unified_kg[1:,0]))}, {len(set(unified_kg[1:,1]))}, {len(set(unified_kg[1:,2]))}')    

gen_unified_kg(args.input_kg1, args.input_kg2, args.output_kg)
print('===Done===')