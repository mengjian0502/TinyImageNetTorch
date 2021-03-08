PYTHON="/home/mengjian/anaconda3/bin/python"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

save_path="./save/resnet18_tinyImgNet/"
log_file="resnet18_tinyImgNet_eval.log"

dataset=tiny_imagenet
lr=0.001
wd=0.0
epochs=15
batch_size=100

$PYTHON train.py --dataset ${dataset} \
    --data_path ./tiny-imagenet-200/ \
    --lr ${lr} \
    --wd ${wd} \
    --batch_size ${batch_size} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --save_path ${save_path} \
