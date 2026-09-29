cd "$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0
# Example SoundSpaces launcher — adjust paths for your environment.
WORLD_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Processing SoundSpaces dataset with $WORLD_SIZE GPU(s)"
python train_av_ss.py -s /workspace/data/soundspaces_speech/ \
    --eval \
    --start_checkpoint "/workspace/SAVAF-AV/output/1" \
    --checkpoint_iterations 2000 \
    --iterations 10000 \
    --checkpoint_path "/workspace/SAVAF-AV/output_soundspaces/1"
    # --av_checkpoint_path "/workspace/SAVAF-AV/output/1/audio_chkpnt10000.pth"
    # --eval_aud
