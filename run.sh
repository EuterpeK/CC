width=448
height=448

for fm in "median"
do
    CUDA_VISIBLE_DEVICES=1 python focal.py \
                    --focal_mode ${fm} \
                    --dataset CO3D \
                    --batch_size 32 \
                    --width ${width} \
                    --height ${height} 
done

