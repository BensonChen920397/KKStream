# KKStream Project 

## Data Path
cfda4 : /tmp2/yhchen/KKStream_project/KKStream_dataset/v4_kg
```
 v4_kg
 ├── all_kg.txt
 ├── tv_triplet.txt
 └── vod_triplet.txt
```
## Make folder
```
mkdir LINE_2nd/dataset/line_kg
cd LINE_2nd/dataset/
unzip v4_kg.zip
cd ..
```
### For WeiLe's experiment
```
mkdir LINE_2nd/code_line/WeiLe/result
mkdir LINE_2nd/code_line/WeiLe/result/embedding
mkdir LINE_2nd/code_line/WeiLe/result/pairs
```
### For YuHsuan's experiment
```
mkdir LINE_2nd/code_line/YuHsuan/result
mkdir LINE_2nd/code_line/nrms-transformer/embedding_result
```

## Description
### For WeiLe's experiment
This is the experiment before 31/08/2022 <br>
* Dataset preprocess 
    * 1: Preprocess ground truth csv
    * 2: Generate unified kg
    * 3: Unified kg to smore input format and mapping txt
* LINE model
    * 1 : Smore-->LINE-->get embeddings
    * 2 : LINE embeddings --> top-N result
    * 3 : Evaluation --> result txt and spectrum plot
* Analysis
    * LINE pairs cosine similarity vs overlap meta or Jacaard
    * LINE union SBert pairs --> Reranked by p*LINE + (1-p)*SBert
### For YuHsuan's experiment
This is the experiment before 30/04/2023 <br>
* LINE model
    * 1 : nrms-transformer-->SBERT-->get SBERT embeddings
    * 2 : Smore-->LINE-->get LINE embeddings (The SBERT embeddings get involved as well)
    * 3 : LINE embeddings -->Evaluation --> result txt
* Analysis
    * The efficacy of negative sample punishment on LINE
    * The efficacy of LINE + SBERT embedding

## Environment
- Step1: Create enviroment with python 3.7
- Step2: Follow the command below 
  - ```pip install torch torchvision torchaudio```
  - ```pip install pandas```
  - ```pip install scipy``` 
  - ```pip install -q transformers==4.7.0 fugashi ipadic```
  - ```pip install sklearn```

## Run example
* Dataset preprocess 
```
cd code_dataset/
sh gen.sh
cd ..
```
### For WeiLe's experiment
* LINE model
```
cd code_line/WeiLe/
sh run_line.sh
sh run_score.sh
sh run_eval.sh
```
### For YuHsuan's experiment
* SBERT model
```
cd code_line/YuHsuan/
sh run_sbert.sh
```
* LINE model
```
cd code_line/YuHsuan/
sh run_line.sh
sh run_eval.sh
```
## Related Documents
* [Slides](https://docs.google.com/presentation/d/1mTO7j8HHy_jf_p1z_BM-y7oEZr6LzpHAJPpv-8MDxFk/edit#slide=id.g218960f85af_1_0)
* [Reports](https://drive.google.com/drive/folders/12v1dc1SHTzzUwGoPuqBF3zFD53B83dOH)

