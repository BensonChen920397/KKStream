#include <stdio.h>
#include <stdlib.h>
#include <string.h> // strcpy

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "../src/model/LINE.h"

using namespace std;

#define _GLIBCXX_USE_CXX11_ABI 1
#define HASH_TABLE_SIZE 30000000


class HashTable {
    public:
        vector< long > table;
        vector< char* > keys;
};

// Key Process
HashTable vertex_hash;

unsigned int BKDRHash(char *key) {
    
    unsigned int seed = 131; // 31 131 1313 13131 131313 etc..
    unsigned int hash = 0;
    while (*key)
    {
        hash = hash * seed + (*key++);
    }
    return (hash % HASH_TABLE_SIZE);
}

long SearchHashTable(HashTable& hash_table, char *key) {

    unsigned int pos = BKDRHash(key);
    while (1)
    {
        if (hash_table.table[pos] == -1)
            return -1;
        if ( !strcmp(key, hash_table.keys[ hash_table.table[pos] ]) )
            return hash_table.table[pos];
        pos = (pos + 1) % HASH_TABLE_SIZE;
    }
}

void LoadPreTrain(string filename, int tar_dim, vector<vector<double>>& given_vec) {

    FILE *fin;
    char c_line[10000];
    char* pch;
    int tok_cnt=0, dim=0, i=0;
    unsigned long long max_line=0;
    cout << "Pretrain Data Loading:" << endl;
    fin = fopen(filename.c_str(), "rb");
    if (fgets(c_line, sizeof(c_line), fin) == NULL)
        return ;
    pch = strtok(c_line," ");
    while(pch != NULL){
        string tmp = pch;
        if(tok_cnt == 0) max_line = atoi(tmp.c_str());
        else dim = atoi(tmp.c_str());
        tok_cnt += 1;
        pch = strtok(NULL," ");
    }
    cout << "\t# of Pre-train:\t\t" << max_line << endl;
    cout << "\tDimensions:\t\t" << dim << endl;
    if (dim != tar_dim){
        cout << "Dimension not matched, Skip Loading Pre-train model.";
        fclose(fin);
    }else{
        while (fgets(c_line, sizeof(c_line), fin)){
            //get each line
            tok_cnt = 0;
            char v[160];
            vector <double> emb;
            pch = strtok(c_line," ");
            //each line processing
            while(pch != NULL){
                string tmp = pch;
                if(tok_cnt == 0) strcpy(v, tmp.c_str());
                else emb.push_back(atof(tmp.c_str()));
                tok_cnt += 1;
                pch = strtok(NULL," ");
            }
            long vid = SearchHashTable(vertex_hash, v);
            if(vid != -1)
            {
                given_vec[vid] = emb;
            }
            else continue;
        }
        fclose(fin);
    }
}


int main()
{


    FILE *fin;
    string filename;


    vector<vector<double>> given_vec_;

    LoadPreTrain("/tmp2/yhchen/KKStream_project/LINE/LINE_2nd/code_line/sbert.embed.txt", 768, given_vec_);


    // char network_file[100], rep_file[100];
    // int dimensions=64, undirected=1, negative_samples=5, sample_times=10, threads=1, order=2;
    // double init_alpha=0.025;

    // if (order==1)
    //     order = 1;
    // else
    //     order = 2;

    // LINE *line;
    // line = new LINE();

    return 0;
}