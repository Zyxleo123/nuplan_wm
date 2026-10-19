###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="/zfsauton/datasets/ArgoRL/brianyan/nuplan/dataset/nuplan-v1.1/splits/trainval"
NUPLAN_MAP_PATH="/zfsauton/datasets/ArgoRL/brianyan/nuplan/dataset/maps"
 # nuplan map path (e.g., "/data/nuplan-v1.1/maps")

TRAIN_SET_PATH="/zfsauton/scratch/mineuih/nuplan_wm/cache" # preprocess training data
###################################

python data_process.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--total_scenarios 1000000 \
--num_workers 32 \
"$@"

