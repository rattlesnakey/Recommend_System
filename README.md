# Setup

```shell
conda env create -f environment.yaml
```



# File Structure

```shell
├── data
│   └── train_small.zip
├── accelerate_config.yaml
├── environment.yaml
├── scripts
│   └── train.sh
└── src
    ├── dataset.py
    ├── model.py
    ├── params.py
    ├── __pycache__
    ├── train.py
    └── utils.py
```



# Train

```shell
cd scripts && bash train.sh
```

```shell
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


accelerate launch --config_file ../accelerate_config.yaml ../src/train.py \
    --batch_size ${BATCH_SIZE} \
    --epoch ${EPOCH} \
    --lr ${LR} \
    --seed ${SEED} \
    --patience ${PATIENCE} \
    --dataset_path ${DATASET_PATH} \
    --model ${MODEL} \
    --ckpt_path ${CKPT_PATH} 2>&1 | tee ${LOG_PATH}/${RUN_NAME}.log

```

# Experiments

> On test dataset



## Settings

### xDeepFM

```python
net = xDeepFM(
	d_embed=8, deep_layers=[256, 128, 64, 32], deep_dropout=0.25,
	split_half=True, cross_layer_sizes=[32, 16, 8],
	n_classes=1
)
```



### ProXDeepFM

```python
net = ProXDeepFM(
	d_embed=8, deep_layers=[256, 128, 64, 32], deep_dropout=0.25,
	split_half=True, cross_layer_sizes=[32, 16, 8],
	reduction_ratio=2, #! SENet
	bilinear_type="field_interaction", #! Bilinear-Interaction
)
```



## Results

```json
{
  'xDeepFM':{
    'AUC':0.7812395413168691,
    'Log_loss':0.4645812096766336,
    }
  'ProXDeepFM':{
  	'AUC':0.7820685198456263,
  	'Log_loss':0.46404934289079625
	}
}
```

