export CUDA_VISIBLE_DEVICES=$1
dataset=$2
workspace=$3

python train.py \
--source_path $dataset -m $workspace \
--eval \
--rand_pcd \
--sh_degree 0 \
--iterations 30000 --position_lr_max_steps 30000 \
--save_iterations 7000 15000 30000 \
--densify_until_iter 10000 \
--densify_grad_threshold 0.0002 \
--prune_threshold 0.01 \
--opacity_reset_interval 1000 \
--gaussiansN 2 \
--coprune --coprune_threshold 3

python render.py \
--source_path $dataset -m $workspace \
--skip_train \
--render_depth
