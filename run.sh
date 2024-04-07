width=448
height=448


for ds in 'SUN3D'
do
    for fm in "weiszfeld" "median" 
    do
        CUDA_VISIBLE_DEVICES=1 python focal.py \
                        --focal_mode ${fm} \
                        --dataset ${ds} \
                        --batch_size 32 \
                        --width ${width} \
                        --height ${height} 
    done
done


