
set -e

BATCH_SIZE=8
GPU=cuda:1

MVTEC_PATH="/home/hyn/work/dataset/AD/mvtec"
VISA_PATH="/home/hyn/work/dataset/AD/Visa"

echo "================================================================"
echo "VisualAD Final Cross-Dataset Evaluation - CLIP"
echo "================================================================"
echo "Batch Size: ${BATCH_SIZE}"
echo "GPU: ${GPU}"
echo "Backbone: ViT-L/14@336px"
echo "Cross-Attention: ENABLED"
echo "Dataset Paths:"
echo "  - MVTec: ${MVTEC_PATH}"
echo "  - VisA:  ${VISA_PATH}"
echo "Epoch Configuration:"
echo "  - MVTec → VisA: 2 epoch"
echo "  - VisA → MVTec: 1 epochs"
echo "================================================================"
echo ""

# ============================================================================
# CLIP 训练测试函数
# ============================================================================
run_clip_experiment() {
    local train_dataset=$1
    local test_dataset=$2
    local train_path=$3
    local test_path=$4
    local epochs=$5

    local exp_name="${train_dataset}_to_${test_dataset}"
    local exp_dir="./experiments/final_clip/${exp_name}"
    local checkpoint_dir="${exp_dir}/checkpoints"
    local result_dir="${exp_dir}/results"

    echo ""
    echo "=========================================="
    echo "Experiment: ${exp_name}"
    echo "Training on: ${train_dataset} (${epochs} epochs)"
    echo "Testing on: ${test_dataset}"
    echo "=========================================="

    # 训练阶段
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training on ${train_dataset}..."
    python train.py \
        --train_data_path ${train_path} \
        --save_path ${checkpoint_dir} \
        --train_dataset ${train_dataset} \
        --backbone "ViT-L/14@336px" \
        --epoch ${epochs} \
        --batch_size ${BATCH_SIZE} \
        --device ${GPU}

    if [ $? -ne 0 ]; then
        echo "❌ Training failed for ${exp_name}"
        return 1
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Training completed!"
    echo ""

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Testing on ${test_dataset} (epoch ${epochs})..."

    python test.py \
        --test_data_path ${test_path} \
        --checkpoint_path ${checkpoint_dir}/epoch_${epochs}.pth \
        --test_dataset ${test_dataset} \
        --save_path ${result_dir}/epoch_${epochs} \
        --device ${GPU}

    if [ $? -eq 0 ]; then
        echo "  ✅ Epoch ${epochs} test completed"
    else
        echo "  ⚠️  Epoch ${epochs} test failed"
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Test completed!"
    echo "Results saved in: ${result_dir}"
    echo ""
}

echo ""
echo "################################################################"
echo "# Experiment 1/2: MVTec → VisA"
echo "################################################################"

run_clip_experiment \
    "mvtec" \
    "visa" \
    "${MVTEC_PATH}" \
    "${VISA_PATH}" \
    2

echo ""
echo "################################################################"
echo "# Experiment 2/2: VisA → MVTec (1 epochs)"
echo "################################################################"

run_clip_experiment \
    "visa" \
    "mvtec" \
    "${VISA_PATH}" \
    "${MVTEC_PATH}" \
    1

# ============================================================================
# 完成
# ============================================================================
echo ""
echo "================================================================"
echo "ALL CLIP EXPERIMENTS COMPLETED!"
echo "================================================================"
