# 任务1运行脚本：传统机器学习方法
echo "========================================"
echo "任务1：传统机器学习方法"
echo "========================================"

# 设置数据目录和保存目录
DATA_DIR="./data"
SAVE_DIR="./checkpoints/task1"

# 创建必要的目录
mkdir -p ${SAVE_DIR}
mkdir -p ./results/task1

# 训练模型
echo ""
echo "开始训练传统机器学习模型..."
python task1_train.py \
    --data_dir ${DATA_DIR} \
    --save_dir ${SAVE_DIR} \
    --models knn softmax svm \
    --seed 42

# 测试模型
echo ""
echo "开始测试模型..."
python task1_test.py \
    --data_dir ${DATA_DIR} \
    --checkpoint_dir ${SAVE_DIR} \
    --models knn softmax svm \
    --seed 42

echo ""
echo "========================================"
echo "任务1完成！"
echo "结果保存在: ${SAVE_DIR}"
echo "========================================"