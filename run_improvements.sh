#!/bin/bash
# Run improvement scripts in order. Assumes dataset/ and best_vit.pth exist.
set -e

DATA_DIR="."
# echo "=== 1. Train ViT with class weights ==="
# python3 train_vit_class_weights.py --epochs 1 --data-dir "$DATA_DIR" --output best_vit\ \(1\).pth

echo ""
echo "=== 2. Train projection head (uses best_vit.pth) ==="
# python3 train_projection_head.py --vit-weights best_vit\ \(1\).pth --epochs 2 --data-dir "$DATA_DIR"

# echo ""
echo "=== 3. Evaluate ensemble ==="
python3 eval_ensemble.py --vit-weights best_vit\ \(1\).pth --proj-weights projection_head.pth --data-dir "$DATA_DIR" --method average

# echo ""
# echo "=== 4. Evaluate multi-prompt CLIP ==="
# python3 eval_clip_multi_prompt.py --data-dir "$DATA_DIR"


echo ""
echo "=== 5. Evaluate failure cases ==="
python3 eval_failure_cases.py --data-dir "$DATA_DIR"