# SEED='1'
# NOISE_FACTOR='10'

for SEED in `seq 2 1 10`
# for SEED in '1'
do
    # for NOISE_FACTOR in '10'
    for NOISE_FACTOR in `seq 1 10 200`
    do
        python3 mb_experiment_noise.py $NOISE_FACTOR $SEED
    done
done