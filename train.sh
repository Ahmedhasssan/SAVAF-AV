export CUDA_VISIBLE_DEVICES=0
# DATA_PATH="/home/ah2288/LP_MipNerF/data/nerf_synthetic/hotdog"
for i in {4..13}; do
    python train.py -s /home/ah2288/AV-3DGS/RWAVS_3DGS_data/release/$i\
        --eval \
        --checkpoint_iteration 30010 \
        --iterations 30010 \
        --checkpoint_path "/home/ah2288/AV-3DGS/output/$i" \
        --start_checkpoint  "/home/ah2288/AV-3DGS/output/$i" \
        # --eval_vision \
        # --model_path "/home/ah2288/3DGS_Original/gaussian-splatting/output/kitchen_full" 
done
    #/home/ah2288/gaussian-splatting/data/360_v2/bicycle --eval  #/home/ah2288/gaussian-splatting/data/tandt/t --eval 