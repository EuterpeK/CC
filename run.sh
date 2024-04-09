width=512
height=384


for ds in 'SUN3D' 'NYUv2' 'Waymo' 'ARKitScenes' 'KITTI' 'ScanNet'
do
    for fm in "weiszfeld" "median"
    do
        CUDA_VISIBLE_DEVICES=0 python pipeline.py \
                        --focal_mode ${fm} \
                        --dataset ${ds} \
                        --batch_size 1 \
                        --width ${width} \
                        --height ${height}  
    done
done


