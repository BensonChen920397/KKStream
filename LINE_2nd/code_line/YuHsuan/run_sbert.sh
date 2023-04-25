with_meta=true
if_pooling=true
fixed_negative=true
negative_type='easy+hard' # 'easy', 'hard', 'easy+hard'

python3 ../nrms-transformer/model/twotower.py \
        ../nrms-transformer/train.pairs \
        ${with_meta} \
        ${if_pooling} \
        ${fixed_negative} \
        ${negative_type}