sp=$1
mu=$2
dir=$3
gpu=$4

rounds=300
epochs=3
total_clients=100
rd_clients=10

python dst.py \
 --distribution noniid --batch-size 20 \
 --dataset cifar10 \
 --lr 0.001 \
 -K ${total_clients} -C ${rd_clients} \
 -R ${rounds} -E ${epochs} \
 --rounds-between-readjustments 10 \
 --sparsity $sp --readjustment-ratio 0.1 \
 --pruning-begin 9 --pruning-interval 3 \
 -d $dir \
 --prox ${mu} \
 -o cifar10-noniid-E${epochs}R${rounds}-prox${mu}-s${sp}r0.1-dir${dir}-10_9_3 --device $gpu
