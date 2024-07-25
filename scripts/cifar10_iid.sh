dir=$1
sp=$2
gpu=$3

mu=0.0
rounds=100
epochs=3
total_clients=10
rd_clients=5

python dst.py \
 --distribution iid --batch-size 20 \
 --dataset cifar10 \
 --lr 0.001 \
 -K ${total_clients} -C ${rd_clients} \
 -R ${rounds} -E ${epochs} \
 --rounds-between-readjustments 10 \
 --sparsity $sp --readjustment-ratio 0.01 \
 --pruning-begin 9 --pruning-interval 2 \
 -d $dir \
 --prox ${mu} \
 -o cifar10-iid-E${epochs}R${rounds}-prox${mu}-s${sp}r0.01-dir${dir} --device $gpu
