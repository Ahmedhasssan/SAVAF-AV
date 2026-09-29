export HIP_VISIBLE_DEVICES=0
DATA_DIR="/workspace/data/release"
OUTPUT_DIR="/workspace/SAVAF-AV/output"
mkdir -p "${OUTPUT_DIR}"

for i in {1..13}; do
    echo "==> Training scene $i"
    python train.py -s ${DATA_DIR}/$i \
        -m "${OUTPUT_DIR}/$i" \
        --eval \
        --iterations 30010 \
        --checkpoint_iterations 30010 \
        --checkpoint_path "${OUTPUT_DIR}/$i"
        # --start_checkpoint "${OUTPUT_DIR}/$i" \
        # --eval_vision
done