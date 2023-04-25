# unified kg generate (output 1 txt)
python3 v4_to_unified_kg.py \
 --input_kg1 ../dataset/v4_kg/vod_triplet.txt \
 --input_kg2 ../dataset/v4_kg/tv_triplet.txt \
 --output_kg ../dataset/v4_kg/all_kg.txt


# convert unified kg to smore format
python3 kg_to_smore.py \
 --input_kg ../dataset/v4_kg/all_kg.txt \
 --output_kg ../dataset/line_kg/line-kg.txt \
 --output_kg_idx ../dataset/line_kg/line-kg.idx.txt

# Preprocess kk labelled ground truth
python3 gt_process.py
