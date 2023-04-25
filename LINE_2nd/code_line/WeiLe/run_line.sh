set -x  # Print commands and their arguments as they are executed.

# to do : download smore package and change the path '/tmp2/weile/smore/cli/line' yourself 

for ep in 500
do
    for DIM in 128
    do
        ../../../smore/cli/line -train ../../dataset/line_kg/line-kg.txt -save result/embedding/kg-2nd-$ep.txt -order 2 -dimensions $DIM -threads 30 -sample_times $ep
    done
done

