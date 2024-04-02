CUDA_VISIBLE_DEVICES=0 python focal.py \
                --focal_mode median \
                --dataset CO3D \
                --batch_size 24 \
                --width 560 \
                --height 448 

CUDA_VISIBLE_DEVICES=0 python focal.py \
                --focal_mode median \
                --dataset NYUv2 \
                --batch_size 24 \
                --width 560 \
                --height 448 

CUDA_VISIBLE_DEVICES=0 python focal.py \
                --focal_mode median \
                --dataset KITTI \
                --batch_size 24 \
                --width 560 \
                --height 448 