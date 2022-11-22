import pandas as pd
import numpy as np 
import datetime 
import sys
from sklearn.model_selection import train_test_split 
import torch 
from torch import nn 
from torch.utils.data import Dataset, DataLoader  
import torch.nn.functional as F 
import torchkeras
from sklearn.preprocessing import LabelEncoder,QuantileTransformer
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from torchkeras import summary
from dataset import *
from model import *
from utils import *
from torchkeras.metrics import AUC
from sklearn.metrics import roc_auc_score, log_loss
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from copy import deepcopy
from torch.optim.lr_scheduler import LambdaLR
from sklearn import metrics
import matplotlib.pyplot as plt
from params import args
import os


def get_dataset(path, device='cuda'):
    #! 读取数据
    dfdata = pd.read_csv(path, sep="\t", header=None)
    
    dfdata.columns = ["label"] + ["I"+str(x) for x in range(1,14)] + [
        "C"+str(x) for x in range(14, 40)]

    cat_cols = [x for x in dfdata.columns if x.startswith('C')]
    num_cols = [x for x in dfdata.columns if x.startswith('I')]
    
    #! SimpleImputer 对缺失值进行填充(默认是 mean 填充)
    #! quantileTransformer 对数据进行非线性映射(采用的是高斯映射)
    num_pipe = Pipeline(steps=[('impute',SimpleImputer()),('quantile',QuantileTransformer())])

    for col in cat_cols:
        #! 给所有类别数据 Transform 成数字的 label
        dfdata[col]  = LabelEncoder().fit_transform(dfdata[col])

    #! 所有的 Number 类型用 Pipeline 进行 preprocess
    dfdata[num_cols] = num_pipe.fit_transform(dfdata[num_cols])

    #! 记录每个的类别特征所包含的类别个数
    categories = [dfdata[col].max()+1 for col in cat_cols]


    #! 划分数据集
    dftrain_val, dftest = train_test_split(dfdata, test_size=0.2)
    dftrain, dfval = train_test_split(dftrain_val, test_size=0.2)

    #! 构建 Dataset
    ds_train = DfDataset(dftrain, device, label_col="label", num_features=num_cols, cat_features=cat_cols,
                        categories=categories, is_training=True)

    ds_val = DfDataset(dfval, device, label_col="label", num_features=num_cols, cat_features=cat_cols,
                        categories=categories, is_training=True)

    ds_test = DfDataset(dftest, device, label_col="label", num_features=num_cols, cat_features=cat_cols,
                        categories=categories, is_training=True)
    return ds_train, ds_val, ds_test


def get_dataloader(ds_train, ds_val, ds_test, batch_size=2048):
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    return dl_train, dl_val, dl_test


def create_net(model, ds_train):
    if model == 'xDeepFM':
        net = xDeepFM(
            d_numerical=ds_train.X_num.shape[1], #! 数值类型特征的个数
            categories=ds_train.get_categories(),#! 类别类型特征的个数
            d_embed=8, deep_layers=[256, 128, 64, 32], deep_dropout=0.25,
            split_half=True, cross_layer_sizes=[32, 16, 8],
            n_classes=1
        )
    elif model == 'DeepFM':
        net = DeepFM(
            d_numerical=ds_train.X_num.shape[1], #! 数值类型特征的个数
            categories=ds_train.get_categories(),#! 类别类型特征的个数
            d_embed=8, deep_layers=[128, 64, 32], deep_dropout=0.25,
            n_classes=1
        )
    elif model == 'ProXDeepFM':
        net = ProXDeepFM(
            d_numerical=ds_train.X_num.shape[1], #! 数值类型特征的个数
            categories=ds_train.get_categories(),#! 类别类型特征的个数
            d_embed=8, deep_layers=[256, 128, 64, 32], deep_dropout=0.25,
            split_half=True, cross_layer_sizes=[32, 16, 8],
            reduction_ratio=2,
            bilinear_type="field_interaction",
            n_classes=1
        )
    return net 


def plot_metric(dfhistory, metric, args):
    train_metrics = dfhistory["train_"+metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()
    plt.savefig(os.path.join(args.ckpt_path, f'{metric}.png'), dpi=100)
    

def get_metrics(model, dl):
    preds = torch.sigmoid(model.predict(dl))
    labels = torch.cat([x[-1] for x in dl])
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    auc = roc_auc_score(labels, preds)
    logloss = log_loss(y_true=labels, y_pred=preds)
    return {'AUC':auc, 'Log-loss':logloss}
    

def main(args):
    set_seed(args.seed)
    #TODO: if DDP 
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], device_placement=False)
    device = accelerator.device
    ds_train, ds_val, ds_test = get_dataset(args.dataset_path, device)
    dl_train, dl_val, dl_test = get_dataloader(ds_train, ds_val, ds_test, batch_size=args.batch_size)
    net = create_net(args.model, ds_train).to(device)

    #! 先取一个 batch 的 features 出来看看 summay
    for features, labels in dl_train:
        break 
    if accelerator.is_local_main_process:
        accelerator.print("net:\n", net)
        accelerator.print(summary(net, input_data=features))

    loss_fn = nn.BCEWithLogitsLoss()
    metrics_dict = {"auc":AUC()}
    
    named_parameters = list(net.named_parameters())
    params = [p for n, p in named_parameters if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=0.001) 
    len_dataset = len(dl_train)
    
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (args.epoch+1))

    #! Accelerator 封装所有加速的元素
    net, dl_train, dl_val, optimizer, scheduler = accelerator.prepare(net, dl_train, dl_val, optimizer, scheduler)
    model = MyPipeline(
                net,
                loss_fn=loss_fn,
                accelerator=accelerator,
                metrics_dict=metrics_dict,
                optimizer=optimizer,
                lr_scheduler=scheduler,
            )   
    os.makedirs(args.ckpt_path, exist_ok=True)
    #! 直接传 dataloader
    dfhistory = model.fit(train_data=dl_train, val_data=dl_val, epochs=args.epoch, patience=args.patience,
                      monitor="val_auc", mode="max", ckpt_path=os.path.join(args.ckpt_path, 'checkpoint.pt')) 

    if accelerator.is_local_main_process:
        #! 可视化 loss 和 auc
        plot_metric(dfhistory, "loss", args)
        plot_metric(dfhistory, "auc", args)

        #! 测试集结果
        #! dl_test 是没有分布式的哈，然后主进程的 model 也是非分布式的
        metrics = get_metrics(model, dl_test)
        accelerator.print(metrics)
        dump_json(os.path.join(args.ckpt_path, 'metrics.json'), metrics)
        printlog("Training Done!", accelerator)
    
if __name__ == "__main__":
    args = args()
    sys.exit(main(args))

