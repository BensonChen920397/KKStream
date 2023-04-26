with_meta=true
if_pooling=true
fixed_negative=true
negative_type='easy+hard' # 'easy', 'hard', 'easy+hard'

cd ../nrms-transformer/model/

python3 ./twotower.py \
        train.pairs \
        ${with_meta} \
        ${if_pooling} \
        ${fixed_negative} \
        ${negative_type}