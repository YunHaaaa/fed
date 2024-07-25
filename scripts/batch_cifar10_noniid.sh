gpu=$1

rounds=300
epochs=3
total_clients=100
rd_clients=10
batch_size=20

for sp in 0.3 0.5 0.6 0.8 0.0
do
for mu in 0.0 1.0
do
for dir in 0.0 0.001 0.005 0.01
do

exp=EXP-cifar10-noniid-S${sp}-MU${mu}-D${dir}-start!
#bash autodl_notify.sh $exp
python dst.py \
 --distribution noniid --batch-size ${batch_size} \
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

done
done
done

