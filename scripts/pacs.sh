dir=$1
sp=$2
gpu=$3
python dst.py \
 --dataset pacs \
 --source photo art_painting cartoon --target sketch \
 --lr 0.001 \
 --clients 3 \
 --rounds 40 --rounds-between-readjustments 10 \
 --sparsity $sp --readjustment-ratio 0.01 \
 --pruning-begin 9 --pruning-interval 2 -E 2 \
 -d $dir \
 -o pacs-Ts-res18-E2R40-feddst-s${sp}r0.01-dir${dir} --device $gpu
