dataset_path=d-nerf/jumpingjacks

# NEW RUN_ID: YYYY-MM-DD-HHMMSS
RUN_ID=$(date +'%Y-%m-%d-%H%M%S')
echo "RUN_ID: $RUN_ID"

CUDA_VISIBLE_DEVICES=0 \
python train_gui.py \
--dataset_path $dataset_path \
--model_path outputs/$dataset_path/$RUN_ID \
--deform_type node \
--node_num 512 \
--is_blender \
--eval \
--gt_alpha_mask_as_scene_mask \
--local_frame \
--W 800 \
--H 800 # \
# --random_bg_color \
# --white_background \
# --gui

