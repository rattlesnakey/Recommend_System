import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
import torch 
from torch import nn 
from torch.utils.data import Dataset,DataLoader  
import torch.nn.functional as F 
import torchkeras 
from sklearn.preprocessing import LabelEncoder,QuantileTransformer
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer 

#! DataFrame转换成torch数据集Dataset, 特征分割成 X_num, X_cat方式
class DfDataset(Dataset):
    def __init__(self, df,
                 device,
                 label_col,
                 num_features,
                 cat_features,
                 categories,
                 is_training=True):
        
        self.X_num = torch.tensor(df[num_features].values).float().to(device) if num_features else None
        self.X_cat = torch.tensor(df[cat_features].values).long().to(device) if cat_features else None
        self.Y = torch.tensor(df[label_col].values).float().to(device) 
        self.categories = categories
        self.is_training = is_training

    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self,index):
        if self.is_training:
            return ((self.X_num[index], self.X_cat[index]), self.Y[index])
        else:
            return (self.X_num[index], self.X_cat[index])
    
    def get_categories(self):
        return self.categories
    

