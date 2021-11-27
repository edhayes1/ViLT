for i in {0..7}
do
    ( CUDA_VISIBLE_DEVICES=$i python3 preprocess.py -i /data/edward/images/dir_00$((i+1))/ -o /data/edward/pretrain/ --fill & )
done
