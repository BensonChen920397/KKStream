## KKStream Project 

## Make folder
```
mkdir dataset/line_kg
mkdir code_line/result
mkdir code_line/result/embedding
mkdir code_line/result/pairs
cd dataset/
unzip v4_kg.zip
cd ..
```

## Description
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

* LINE model
```
cd code_line/
sh run_line.sh
sh run_score.sh
sh run_eval.sh
```
