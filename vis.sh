width=640
height=480


for ds in 'SUN3D'
do
    for fm in "median"
    do
        CUDA_VISIBLE_DEVICES=0 python distribution.py \
                        --focal_mode ${fm} \
                        --dataset ${ds} \
                        --batch_size 1 \
                        --width ${width} \
                        --height ${height}  
    done
done


