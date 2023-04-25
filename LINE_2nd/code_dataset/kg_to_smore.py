# Convert unified kg to smore format
import numpy as np
import csv
import argparse

parser=argparse.ArgumentParser(description='line kg generator')
parser.add_argument('--input_kg', type=str, help='input kg that need to be converted to smore format')
parser.add_argument('--output_kg', type=str, help='output converted kg')
parser.add_argument('--flag_output_kg_idx', type=int, default=1,
                    help='1: save index map ; 0: not save index map')
parser.add_argument('--output_kg_idx', type=str, help='output converted kg index map')
args=parser.parse_args()
# read kg
kg_all = []
f = open(args.input_kg)
lines = f.readlines()
for i in lines[1:]:
    kg_all.append(i.strip('\n').split('\t'))
kg_all = np.array(kg_all)

# kg statistics
print('===kg statistics===')
print(f'shape of the kg: {kg_all.shape}')
for i in ['artist', 'genre', 'rating', 'type']:
    print(f'edge of {i}: {len(kg_all[kg_all[:,1]==i])}  ;  unique ent of {i}: {len(set(kg_all[kg_all[:,1]==i][:,2]))}' )

# encode title and meta
print('\nEncoding title and meta...')
tit_to_idx, meta_to_idx = {}, {}
count_t, count_m = 0, 0
for i in range(len(kg_all)):
    # encode title
    if kg_all[i][0] not in tit_to_idx:
        tit_to_idx[kg_all[i][0]] = 'e' + str(count_t)
        count_t+=1
    kg_all[i][0] = tit_to_idx[kg_all[i][0]]

    # encode meta
    if kg_all[i][2] not in meta_to_idx:
        meta_to_idx[kg_all[i][2]] = 'm' + str(count_m)
        count_m+=1
    kg_all[i][2] = meta_to_idx[kg_all[i][2]]    

print(f'# of encoded title: {count_t}')
print(f'# of encoded meta: {count_m}')

# kg in smore (line) format
train = []
for i in kg_all:
    train.append([i[0],i[2],1])
# for i in kg_all: # lower weight for genre, type, rating
#     if i[1]!= 'artist':
#         train.append([i[0],i[2],0.5])
#     else:
#         train.append([i[0],i[2],1])

# index map
idx_map = []
for i in tit_to_idx:
    idx_map.append([i, tit_to_idx[i]])
for i in meta_to_idx:
    idx_map.append([i, meta_to_idx[i]])

# save
with open(f"{args.output_kg}", "w", newline="") as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(train)
    
with open(f"{args.output_kg_idx}", "w", newline="") as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(idx_map)    