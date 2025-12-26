echo "========================================"
echo "任务2"
echo "========================================"

DATA_DIR="./data"
SAVE_DIR="./checkpoints/task2/largeresnet"
MODEL="largeresnet"
EPOCHS=200                     
BATCH_SIZE=32
LR=0.01
OPTIMIZER="sgd"
LR_SCHEDULER="cosine"
LABEL_SMOOTHING=0.1
EARLY_STOPPING_PATIENCE=15      
WEIGHT_DECAY=5e-4
IMAGE_SIZE=224

mkdir -p ${SAVE_DIR}

if command -v nvidia-smi &> /dev/null
then
    CUDA_FLAG=""
else
    CUDA_FLAG="--no_cuda"
fi

echo ""
echo "训练配置:"
echo "  模型: ${MODEL}"
echo "  最大训练轮数: ${EPOCHS}"
echo "  早停耐心值: ${EARLY_STOPPING_PATIENCE} epochs"
echo "  优化器: ${OPTIMIZER}"
echo "  初始学习率: ${LR}"
echo "  学习率调度: ${LR_SCHEDULER}"
echo "  权重衰减: ${WEIGHT_DECAY}"
echo "  标签平滑: ${LABEL_SMOOTHING}"
echo ""

python task2_train.py \
    --data_dir ${DATA_DIR} \
    --save_dir ${SAVE_DIR} \
    --model ${MODEL} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --optimizer ${OPTIMIZER} \
    --weight_decay ${WEIGHT_DECAY} \
    --lr_scheduler ${LR_SCHEDULER} \
    --label_smoothing ${LABEL_SMOOTHING} \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    --image_size ${IMAGE_SIZE} \
    --val_split 0.1 \
    --num_workers 4 \
    ${CUDA_FLAG} \
    --seed 42

python task2_test.py \
    --data_dir ${DATA_DIR} \
    --checkpoint_dir ${SAVE_DIR} \
    --model ${MODEL} \
    --batch_size ${BATCH_SIZE} \
    --image_size ${IMAGE_SIZE} \
    --num_workers 4 \
    ${CUDA_FLAG} \
    --seed 42

echo ""
echo "========================================"
echo "完成！"
echo "========================================"