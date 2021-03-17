PYTHON="/home/mengjian/anaconda3/envs/myenv_pc/bin/python"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

model=resnet18_imagenet_quant
wbit=4
abit=4
mode=mean
k=4

pretrained_model="./save/resnet18_imagenet_quant_w4_a4_mode_mean_k4_lambda_wd0.0001_swpFalse/model_best.pth.tar"
save_path="./save/resnet18_tinyImgNet_w${wbit}_a${abit}/"
log_file="resnet18_tinyImgNet_w${wbit}_a${abit}.log"

dataset=tiny_imagenet
lr=0.0005
wd=0.0
epochs=15
batch_size=60

$PYTHON -W ignore train.py --dataset ${dataset} \
    --model ${model} \
    --data_path ./tiny-imagenet-200/ \
    --lr ${lr} \
    --wd ${wd} \
    --batch_size ${batch_size} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --save_path ${save_path} \
    --wbit ${wbit} \
    --abit ${abit} \
    --clp \
    --a_lambda ${wd} \
    --resume ${pretrained_model} \
    --fine_tune \
    --q_mode ${mode} \
    --k ${k} \
