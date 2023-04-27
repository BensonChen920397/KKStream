set -x  # Print commands and their arguments as they are executed.

# to do : download smore package and change the path '/tmp2/weile/smore/cli/line' yourself 

for ep in 10000
do
    for DIM in 128
    do
        ../../../smore/cli/line \
        -train ../../dataset/line_kg/line-kg.txt \
        -save result/kg-2nd-$ep.txt \
        -order 2 \
        -dimensions $DIM \
        -threads 30 \
        -sample_times $ep \
        -negative_samples 10 \
        -pretrain ../../nrms-transformer/embedding_result/fourtower.embed \
        -beta 0.3
    done
done

