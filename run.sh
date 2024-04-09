width=512
height=384

# 'ARKitScenes' 
for ds in 'ScanNet'
do
    for fm in "median"
    do
        CUDA_VISIBLE_DEVICES=1 python pipeline.py \
                        --focal_mode ${fm} \
                        --dataset ${ds} \
                        --batch_size 100 \
                        --width ${width} \
                        --height ${height}  
    done
done


