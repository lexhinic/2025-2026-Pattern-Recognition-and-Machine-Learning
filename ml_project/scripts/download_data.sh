DATA_URL="https://cloud.tsinghua.edu.cn/f/72aab178f61948c095dd/?dl=1"
DATA_DIR="./data"
TEMP_DIR="./temp_download"

# 创建临时目录
mkdir -p ${TEMP_DIR}

echo ""
echo "正在下载数据集..."
echo "下载地址: ${DATA_URL}"

# 使用wget下载
if command -v wget &> /dev/null; then
    wget -O ${TEMP_DIR}/cub200_dataset.zip "${DATA_URL}"
    DOWNLOAD_STATUS=$?
# 使用curl下载
elif command -v curl &> /dev/null; then
    curl -L -o ${TEMP_DIR}/cub200_dataset.zip "${DATA_URL}"
    DOWNLOAD_STATUS=$?
else
    echo "错误: 系统中未找到wget或curl命令"
    exit 1
fi

# 检查下载是否成功
if [ ${DOWNLOAD_STATUS} -ne 0 ]; then
    echo "下载失败！"
    exit 1
fi

echo "下载完成！"

# 解压数据集
echo ""
echo "正在解压数据集..."

if command -v unzip &> /dev/null; then
    unzip -q ${TEMP_DIR}/cub200_dataset.zip -d ${TEMP_DIR}
    
    if [ -d "${TEMP_DIR}/data" ]; then
        mv ${TEMP_DIR}/data/* ${DATA_DIR}/
    else
        mkdir -p ${DATA_DIR}
        mv ${TEMP_DIR}/train ${DATA_DIR}/
        mv ${TEMP_DIR}/val ${DATA_DIR}/
    fi
    
    echo "解压完成！"
else
    echo "错误: 系统中未找到unzip命令"
    exit 1
fi

# 清理临时文件
echo ""
echo "清理临时文件..."
rm -rf ${TEMP_DIR}

# 验证数据集
echo ""
echo "验证数据集..."
if [ -d "${DATA_DIR}/train" ] && [ -d "${DATA_DIR}/val" ]; then
    TRAIN_CLASSES=$(ls -d ${DATA_DIR}/train/*/ | wc -l)
    VAL_CLASSES=$(ls -d ${DATA_DIR}/val/*/ | wc -l)
    
    echo "训练集类别数: ${TRAIN_CLASSES}"
    echo "验证集类别数: ${VAL_CLASSES}"
    echo ""
    echo "数据集准备完成！"
else
    echo "错误: 数据集结构不正确"
    exit 1
fi

echo ""
echo "数据集下载和解压完成！"