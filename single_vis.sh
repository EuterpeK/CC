width=640
height=480

# 'ARKitScenes' 
for ds in 'SUN3D'
do
    for fm in "median"
    do
        CUDA_VISIBLE_DEVICES=0 python single_vis.py \
                        --focal_mode ${fm} \
                        --dataset ${ds} \
                        --batch_size 100 \
                        --width ${width} \
                        --height ${height}  \
                        --img_name box.jpg
    done
done

