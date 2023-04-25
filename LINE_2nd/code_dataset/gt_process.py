import numpy as np
import csv
import pickle

f = open('../dataset/labelled_gt/shuffled_pruned_3-10_dup_pairs_all_hirdawang_20220609.csv', 'r')
L = []
lines = f.readlines()
for line in lines[1:]:
    l = line.strip('\n').rstrip(',').split(',')
    L.append(l)
L=np.array(L)
L = L[:,[1,2,3]]
for i in range(len(L)):
    L[i][0], L[i][1] = eval(L[i][0]), eval(L[i][1])

print('Seperate data according to its label...')
# 1: notsame,  2: similar,  3: same  , 4: not sure 
notsame = set([tuple(i) for i in L[L[:,2]=='1'][:,0:2]])
similar = set([tuple(i) for i in L[L[:,2]=='2'][:,0:2]])
same = set([tuple(i) for i in L[L[:,2]=='3'][:,0:2]])

print(f'not same pairs: {len(notsame)}')
print(f'not similar pairs: {len(similar)}')
print(f'same pairs: {len(same)}')

#save
with open('../dataset/labelled_gt/kk_notsame.pkl', 'wb') as fp:
    pickle.dump(notsame, fp)
with open('../dataset/labelled_gt/kk_similar.pkl', 'wb') as fp:
    pickle.dump(similar, fp)
with open('../dataset/labelled_gt/kk_same.pkl', 'wb') as fp:
    pickle.dump(same, fp)

