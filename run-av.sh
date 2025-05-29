export CUDA_VISIBLE_DEVICES=0
DATA_DIR="/share/seo/RWAVS_3DGS_data/release/1"
LOG_DIR="./logs"
EXP_NAME="reproduce"
METHOD="anerf"
GS_PATH="/home/ah2288/AV-3DGS/output/1_output"
BACKGROUND=False
# for i in {1..13}
# do
#     echo $i
#     python main.py --data-root ${DATA_DIR}/$i/ --log-dir ${LOG_DIR}/$i/audio_output/ --output-dir $METHOD/$EXP_NAME/ --conv --lr 5e-4 --max-epoch 100
# done

python av-main.py --data-root ${DATA_DIR}/ --model-path ${GS_PATH}/ --source-path ${DATA_DIR}/ --white-background ${BACKGROUND}/ --log-dir ${LOG_DIR}/$i/audio_output/ --output-dir $METHOD/$EXP_NAME/ --conv --lr 5e-4 --max-epoch 100

# python eval.py --log-dir ${LOG_DIR}/ --output-dir $METHOD/$EXP_NAME/