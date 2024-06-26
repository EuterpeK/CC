width=640
height=480

# 'ARKitScenes' 
for ds in 'NUScenes'
do
    for fm in "median"
    do
        CUDA_VISIBLE_DEVICES=0 python pipeline.py \
                        --focal_mode ${fm} \
                        --dataset ${ds} \
                        --batch_size 100 \
                        --width ${width} \
                        --height ${height}  
    done
done

