for model in line sbert
do
  python pair_scorer.py \
    --input_idx ../../dataset/line_kg/line-kg.idx.txt \
    --input_line_emb result/embedding/kg-2nd-500.txt \
    --Model $model \
    --topN 50 \
    --output_path result/pairs/$model-pairs-top50.txt

    sort -k3gr -o result/pairs/$model-pairs-top50.txt result/pairs/$model-pairs-top50.txt # sort third column
done