gpu=$1

rounds=400
epochs=10
total_clients=100
rd_clients=20
for sp in 0.3 0.5 0.6 0.8 0.0
do
for mu in 0.0 1.0
do
for dir in 0.0 0.001 0.005 0.01
do

exp=EXP-mnist-noniid-S${sp}-MU${mu}-D${dir}-start!
#bash autodl_notify.sh $exp
python dst.py \
 --distribution noniid --batch-size 20 \
 --dataset mnist \
 --lr 0.01 \
 -K ${total_clients} -C ${rd_clients} \
 -R ${rounds} -E ${epochs} \
 --rounds-between-readjustments 10 \
 --sparsity $sp --readjustment-ratio 0.5 \
 --pruning-begin 9 --pruning-interval 10 \
 -d $dir \
 --prox ${mu} \
 -o mnist-noniid-E${epochs}R${rounds}-prox${mu}-s${sp}r0.5-dir${dir}-10_9_10 --device $gpu

done
done
done

