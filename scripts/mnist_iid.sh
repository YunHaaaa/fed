sp=$1
mu=$2
dir=$3
gpu=$4
r=0.5

python dst.py \
 --dataset mnist \
 --distribution iid \
 --lr 0.01 \
 -K 400 -C 20 \
 -R 400 -E 10 \
 --rounds-between-readjustments 10 \
 --sparsity $sp --readjustment-ratio ${r} \
 --pruning-begin 9 --pruning-interval 10 \
 -d $dir \
 --prox ${mu} \
 -o mnist-${dist}-K400C20-E10R400-prox${mu}-s${sp}r${r}-dir${dir} --device $gpu

