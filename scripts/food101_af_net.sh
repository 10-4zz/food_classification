python main.py \
    --cfg ./cfgs/af_net.yaml \
    --data_path /home/zxy/code/dataset/Food-101/ \
    --model_name af_net \
    --dataste_name ETHZFOOD101 \
    --batch_size 32 \
    --image_size 224 \
    --output_dir ./output/food101/ \
    --tag food101_af_net