export CUDA_VISIBLE_DEVICES=$1
dataset=$2
workspace=$3

python train.py \
--source_path $dataset -m $workspace \
--eval \
--rand_pcd \
--sh_degree 1 \
--iterations 20000 --position_lr_max_steps 20000 \
--densify_until_iter 10000 \
--densify_grad_threshold 0.0002 \
--gaussiansN 2 \
--coprune --coprune_threshold 5

python render.py \
--source_path $dataset -m $workspace \
--render_depth

python metrics.py \
-s $dataset -m $workspace
