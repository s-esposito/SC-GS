DATASET_PATH=d-nerf/jumpingjacks
# NEW RUN_ID: YYYY-MM-DD-HHMMSS
RUN_ID="2024-11-18-110151"
echo "RUN_ID: $RUN_ID"

CUDA_VISIBLE_DEVICES=0 \
python convert_to_dyn_3dgs.py \
--dataset_path $DATASET_PATH \
--run_path outputs/$DATASET_PATH/$RUN_ID \
--deform_type node \
--node_num 512 \
--is_blender
