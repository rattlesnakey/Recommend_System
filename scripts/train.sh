set -ev

BATCH_SIZE=512
EPOCH=2
LR=0.01
SEED=3407
MODEL=xDeepFM
DATASET_PATH=../data/train_small.zip
PATIENCE=5
RUN_NAME=model=${MODEL}_batch-size=${BATCH_SIZE}_epoch=${EPOCH}_lr=${LR}
CKPT_PATH=../checkpoints/${RUN_NAME}
LOG_PATH=../logs
DEVICES=1,3,4,5
export CUDA_VISIBLE_DEVICES=${DEVICES}

mkdir -p ${CKPT_PATH}
mkdir -p ${LOG_PATH}

# python -u
accelerate launch --config_file ../accelerate_config.yaml ../src/train.py \
    --batch_size ${BATCH_SIZE} \
    --epoch ${EPOCH} \
    --lr ${LR} \
    --seed ${SEED} \
    --patience ${PATIENCE} \
    --dataset_path ${DATASET_PATH} \
    --model ${MODEL} \
    --ckpt_path ${CKPT_PATH} 2>&1 | tee ${LOG_PATH}/${RUN_NAME}.log
