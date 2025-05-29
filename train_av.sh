export CUDA_VISIBLE_DEVICES=0
# DATA_PATH="/home/ah2288/LP_MipNerF/data/nerf_synthetic/hotdog"
WORLD_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) 
for i in {12..12}; do
    # torchrun --nproc_per_node=$WORLD_SIZE --master_port=29505 main_mvsplat.py -s /home/ah2288/AV-3DGS/RWAVS_3DGS_data/release/$i\
    echo "Processing dataset $i with $WORLD_SIZE GPUs"
    python train_av.py -s /home/ah2288/AV-3DGS/RWAVS_3DGS_data/release/$i\
        --eval \
        --start_checkpoint  "/home/ah2288/AV-3DGS/output/$i" \
        --checkpoint_iterations 2000 \
        --iterations 10000 \
        --checkpoint_path "/home/ah2288/AV-3DGS/output/$i" \
        --av_checkpoint_path "/home/ah2288/AV-3DGS/output/$i/audio_chkpnt10000.pth" \
        --eval_aud
done