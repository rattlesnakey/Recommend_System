# -*- encoding:utf-8 -*-
import torch 
from torch import nn
from torch import nn,Tensor 
import torch.nn.functional as F 
from itertools import combinations
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs


from copy import deepcopy
from utils import *
import sys
import pandas as pd

#! 连续值的 Embedding 映射
class NumEmbedding(nn.Module):
    """
    连续特征用linear层编码
    输入shape: [batch_size,features_num(n), d_in], # d_in 通常是1 #! 因为每个 feature field 里只有一个 value
    输出shape: [batch_size,features_num(n), d_out]
    """
    
    def __init__(self, n: int, d_in: int, d_out: int, bias: bool = False) -> None:
        super().__init__()
        #! 这边是创建每个 feature 所对应的 embedding 矩阵
        #! n 是 feature field 的数量
        self.weight = nn.Parameter(Tensor(n, d_in, d_out))
        self.bias = nn.Parameter(Tensor(n, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n):
                #! linear 是一个 1x10 的矩阵，然后对应的 embedding 就是等于 1x1 * 1x10 = 1x10 来得到的
                layer = nn.Linear(d_in, d_out) 
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x_num):
        assert x_num.ndim == 3
        #! batch 矩阵乘法，本质上是 fi x ij -> fj
        x = torch.einsum("bfi,fij->bfj", x_num, self.weight)
        if self.bias is not None:
            x = x + self.bias[None]
        return x
    
class CatEmbedding(nn.Module):
    """
    离散特征用Embedding层编码
    输入shape: [batch_size,features_num], 
    输出shape: [batch_size,features_num, d_embed]
    """
    def __init__(self, categories, d_embed):
        super().__init__()
        #! 所有可能的类别取值，比如性别的 male 和 female 的 id 为 0 和 1，销量的高和低的 id 就是 2 和 3
        #! 这里的 categories 里面的值是每个 feature 类别到底有几个类别，比如性别有两个类别，销量有 3 个类别这样，所以 sum 起来就是每个类别值一个 idx
        self.embedding = torch.nn.Embedding(sum(categories), d_embed)
        #! 这个就是去记录每个类别在 embedding 矩阵里面的起始 idx 
       
        self.offsets = nn.Parameter(
                torch.tensor([0] + categories[:-1]).cumsum(0), requires_grad=False) 
        # self.offsets = torch.tensor([0] + categories[:-1]).cumsum(0) #, requires_grad=False
            
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x_cat):
        """
        :param x_cat: Long tensor of size ``(batch_size, features_num)``
        """
        #! offsets 记录的每个 category feature field 在 embedding matrix 中的起始 index
        x = x_cat + self.offsets[None]
        return self.embedding(x) 
    
    
class CatLinear(nn.Module):
    """
    离散特征用Embedding实现线性层（等价于先F.onehot再nn.Linear()）
    输入shape: [batch_size,features_num], 
    输出shape: [batch_size,features_num, d_out]
    """
    def __init__(self, categories, d_out=1):
        super().__init__()
        self.fc = nn.Embedding(sum(categories), d_out)
        self.bias = nn.Parameter(torch.zeros((d_out,)))
        
        self.offsets = nn.Parameter(
                torch.tensor([0] + categories[:-1]).cumsum(0), requires_grad=False) 
        # self.offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
        
    def forward(self, x_cat):
        """
        :param x: Long tensor of size ``(batch_size, features_num)``
        """
        x = x_cat + self.offsets[None]
        #! 这边就是沿着 embedding 维度加起来，所以得到的是 batch_size 个 scalar 了
        #! 这边这个 Embedding 其实本质上就是把一个 离散值映射成一个 float 值，然后把所有的 category 对应的 float 值进行加和起来
        return torch.sum(self.fc(x), dim=1) + self.bias 
    
    
#! 特征交互 Layer
class FMLayer(nn.Module):
    """
    FM交互项
    """

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x): 
        """
        :param x: Float tensor of size ``(batch_size, num_features, k)``
        """
        
        #! 因为前面得到 Num 和 Cat 的 embedding 的时候，就已经是用当前的 scalar * vector weight 得到的结果了
        #! 所以这里的x是公式中的 <v_i> * xi 
        #! 这边是沿着 feature_num 维度加和，就是每个 feature 的 embedding 加和起来，然后得到 (batch, k), 然后里面每个元素是平方后的
        #! 这边先加就是 先把 <v_i> * xi + <v_j> * vj ... 加起来，然后再平方
        square_of_sum = torch.sum(x, dim=1) ** 2
        #! 这个是先 square 然后再 sum，也是得到 batch, k
        #! 这里先平方就是先得到 <v_i> * <v_i> * xi * xi，然后再加
        sum_of_square = torch.sum(x ** 2, dim=1)
        #! 两个相减刚好就得到他们中间的交叉项目
        ix = square_of_sum - sum_of_square
        #! 看如果是二分类
        if self.reduce_sum:
            #! 得到 (batch, 1)
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix
    


#! Deep部分
class MultiLayerPerceptron(nn.Module):

    def __init__(self, d_in, d_layers, dropout, 
                 d_out = 1):
        super().__init__()
        layers = []
        for d in d_layers:
            layers.append(nn.Linear(d_in, d))
            layers.append(nn.BatchNorm1d(d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            d_in = d
        layers.append(nn.Linear(d_layers[-1], d_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, d_in)``
        """
        return self.mlp(x)
    
    
class DeepFM(nn.Module):
    """
    DeepFM模型。
    """

    def __init__(self, d_numerical, categories, d_embed,
                 deep_layers, deep_dropout,
                 n_classes=1):
        
        super().__init__()
        if d_numerical is None:
            d_numerical = 0
        if categories is None:
            categories = []
            
        self.categories = categories
        self.n_classes = n_classes
        
        self.num_linear = nn.Linear(d_numerical, n_classes) if d_numerical else None
        self.cat_linear = CatLinear(categories, n_classes) if categories else None
        
        #! 这边是得到 embedding
        self.num_embedding = NumEmbedding(d_numerical, 1, d_embed) if d_numerical else None
        self.cat_embedding = CatEmbedding(categories, d_embed) if categories else None
        
        if n_classes == 1:
            self.fm = FMLayer(reduce_sum=True)
            self.fm_linear = None
        else:
            assert n_classes >=2
            #! 根据最后预测的类别的数量来进行判断
            #! 如果是多分类，最后就返回 embedding 然后再用一个 linear 过到类别数量
            self.fm = FMLayer(reduce_sum=False)
            self.fm_linear = nn.Linear(d_embed, n_classes)
        
        #! 这里要把所有的输入都拼接起来，所以要构建一个对应大小的 linear
        self.deep_in = d_numerical*d_embed+len(categories)*d_embed
        
        self.deep = MultiLayerPerceptron(
            d_in=self.deep_in,
            d_layers=deep_layers,
            dropout=deep_dropout,
            d_out=n_classes
        )
        

    def forward(self, x):
        
        """
        x_num: numerical features
        x_cat: category features
        """
        x_num, x_cat = x
        #! 这边 linear 的部分就是直接把原来的特征过一个 linear 映射到 class 类别上面
        x = 0.0
        if self.num_linear:
            #! 这边是 返回的是 class 的数量，class 的数量应该都是 1，所以就可以直接相加了
            x = x + self.num_linear(x_num) 
        if self.cat_linear:
            #! 感觉可以 num_linear 直接返回的 x 直接给这边用
            x = x + self.cat_linear(x_cat)
        
        #! FM部分
        #! 把 num 和 cat 的 embedding append 起来，然后再 cat 
        x_embedding = []
        if self.num_embedding:
            x_embedding.append(self.num_embedding(x_num[..., None]))
        if self.cat_embedding:
            x_embedding.append(self.cat_embedding(x_cat))
        x_embedding = torch.cat(x_embedding,dim=1)
        
        #! 把特征交互后映射到的 class prob 加到原来 linear 的部分
        if self.n_classes == 1:
            x = x + self.fm(x_embedding)
        else: 
            x = x + self.fm_linear(self.fm(x_embedding)) 
            
        #! deep部分
        #! 把 FM 得到的 embedding flatten 平铺开
        x = x + self.deep(x_embedding.view(-1, self.deep_in))
        #! 压缩成一个 scalar prob
        if self.n_classes == 1:
            x = x.squeeze(-1)
        
        return x
    
#! CIN
class CompressedInteractionNetwork(torch.nn.Module):
    #! input_dim 是 feature_field 的个数
    #! cross_layer_size=(256, 128), 是一个 list 或者一个 tuple
    def __init__(self, input_dim, cross_layer_sizes=[256, 128], split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        #! 卷积提取的时候，中间层的卷积的通道数要不要减半
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            #! 卷积的 in_channels 是 feature_fields x prev_dim
            #! out_channel 是 cross_layer_size 
            #! kernel size 是 1 x 1，相当于是一个线性 Linear (对每个 feature_field 的 embedding 的 linear )
            #! conv1d 的 in_channel 就是 cross_feature 的 num 了
            #! out_channel 就是返回多少个 cross_feature 了，因为是 1x1 的 conv1d，所以 embedding 还是一样的
            #! 这边是在 embedding 维度上进行滑动，相当于是把每个 feature_field 的 embedding 里面相同的元素进行卷积
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            
            prev_dim = cross_layer_size
            
            #! fc_input_dim 是最后的输入维度，最后的输入维度是所有中间层卷积的 out_channel 数 concat
            fc_input_dim += prev_dim
        
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)`` 
        """
        xs = list()
        #! x0: (batch_size, num_fields, 1, embed_dim)
        x0, h = x.unsqueeze(2), x 
        #! 就是有几个卷积
        for i in range(self.num_layers):
            #! (batch_size, num_fields, 1, embed_dim) x (batch_size, 1, num_fields, embed_dim)
            #! 这边的 x0 一周都是原来的 x0，h 是后面一直更新得到的
            x = x0 * h.unsqueeze(1)
            #! (batch_size, num_fields, num_fields, embed_dim)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            
            #! 这边就是得到所有交互后的新的 feature embedding
            #! batch_size, cross_features_num, embed_dim
            #! 注意，这边的 channel 不是 embedding 和 NLP 不一样，现在 channel 是 cross_feature_num, 相当于原来的 embedding_dim 维度变成 length 维度了
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            
            #! 在这些 feature embedding 上面做卷积，然后再激活
            x = F.relu(self.conv_layers[i](x))
            
            if self.split_half and i != self.num_layers - 1:
                #! split_half 的话，就分成两波
                #! 分成两波的话，cross_feature 的矩阵就是 (batch_size, num_fields, num_fields/2, embed_dim)
                #! 所以 view 的时候对应的 in_channel 数量就会变少
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            #! 保留所有的 x 
            xs.append(x)
        #! 所有
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))
    
class xDeepFM(nn.Module):
    """
    xDeepFM模型。
    """

    def __init__(self, d_numerical, categories, d_embed,
                 deep_layers, deep_dropout, split_half=True, 
                 cross_layer_sizes=[256, 128],
                 n_classes=1):
        
        super().__init__()
        if d_numerical is None:
            d_numerical = 0
        if categories is None:
            categories = []
            
        self.categories = categories
        self.n_classes = n_classes
        
        self.num_linear = nn.Linear(d_numerical, n_classes) if d_numerical else None
        self.cat_linear = CatLinear(categories, n_classes) if categories else None
        
        #! 这边是得到 embedding
        self.num_embedding = NumEmbedding(d_numerical, 1, d_embed) if d_numerical else None
        self.cat_embedding = CatEmbedding(categories, d_embed) if categories else None
        
        #! CIN 
        self.cin = CompressedInteractionNetwork(len(self.categories) + d_numerical, cross_layer_sizes, split_half)
        
        #! 把 category embedding 直接和 number features 拼接
        self.deep_in = d_numerical*d_embed+len(categories)*d_embed
        
        self.deep = MultiLayerPerceptron(
            d_in=self.deep_in,
            d_layers=deep_layers,
            dropout=deep_dropout,
            d_out=n_classes
        )
        

    def forward(self, x):
        
        """
        x_num: numerical features
        x_cat: category features
        """
        x_num, x_cat = x
        #!linear部分
        #! 这边 linear 的部分就是直接把原来的特征过一个 linear 映射到 class 类别上面
        x = 0.0
        if self.num_linear:
            x = x + self.num_linear(x_num) 
        if self.cat_linear:
            x = x + self.cat_linear(x_cat)
        x_embedding = []
        if self.num_embedding:
            x_embedding.append(self.num_embedding(x_num[..., None]))
        if self.cat_embedding:
            x_embedding.append(self.cat_embedding(x_cat))
        x_embedding = torch.cat(x_embedding,dim=1)

        #! CIN部分
        x += self.cin(x_embedding)
                
        #! deep部分
        x = x + self.deep(x_embedding.view(-1, self.deep_in))
        #! 压缩成一个 scalar prob
        if self.n_classes == 1:
            x = x.squeeze(-1)
        
        return x


class SENetAttention(nn.Module):
    """
    Squeeze-and-Excitation Attention
    输入shape: [batch_size, num_fields, d_embed]   #num_fields即num_features
    输出shape: [batch_size, num_fields, d_embed]
    """
    def __init__(self, num_fields, reduction_ratio=3):
        super().__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.excitation = nn.Sequential(nn.Linear(num_fields, reduced_size, bias=False),
                                        nn.ReLU(),
                                        nn.Linear(reduced_size, num_fields, bias=False),
                                        nn.ReLU())

    def forward(self, x):
        Z = torch.mean(x, dim=-1, out=None) #! 1,Sequeeze
        A = self.excitation(Z) #! 2,Excitation
        V = x * A.unsqueeze(-1) #! 3,Re-Weight
        return V
    
class BilinearInteraction(nn.Module):
    """
    双线性FFM
    输入shape: [batch_size, num_fields, d_embed] #num_fields即num_features
    输出shape: [batch_size, num_fields*(num_fields-1)/2, d_embed]
    """
    def __init__(self, num_fields, d_embed, bilinear_type="field_interaction"):
        super().__init__()
        self.bilinear_type = bilinear_type
        #! 所有的都共享一个 linear
        if self.bilinear_type == "field_all":
            self.bilinear_layer = nn.Linear(d_embed, d_embed, bias=False)
            
        #! 每个 feature 一个 linear
        elif self.bilinear_type == "field_each":
            self.bilinear_layer = nn.ModuleList([nn.Linear(d_embed, d_embed, bias=False)
                                                 for i in range(num_fields)])
        #! 排列组合，每两个 feature 用一个 linear
        elif self.bilinear_type == "field_interaction":
            self.bilinear_layer = nn.ModuleList([nn.Linear(d_embed, d_embed, bias=False)
                                                 for i, j in combinations(range(num_fields), 2)])
        else:
            raise NotImplementedError()

    def forward(self, feature_emb):
        feature_emb_list = torch.split(feature_emb, 1, dim=1)
        if self.bilinear_type == "field_all":
            #! 两两组合不同的 embedding
            #! 这边做的是答哈玛积
            bilinear_list = [self.bilinear_layer(v_i) * v_j
                             for v_i, v_j in combinations(feature_emb_list, 2)]
        elif self.bilinear_type == "field_each":
            bilinear_list = [self.bilinear_layer[i](feature_emb_list[i]) * feature_emb_list[j]
                             for i, j in combinations(range(len(feature_emb_list)), 2)]
        elif self.bilinear_type == "field_interaction":
            #! 先要取出对应的 idx, 然后v0 是 vi, v1 是 vj
            bilinear_list = [self.bilinear_layer[i](v[0]) * v[1]
                             for i, v in enumerate(combinations(feature_emb_list, 2))]
        return torch.cat(bilinear_list, dim=1)


class ProXDeepFM(nn.Module):
    """
    xDeepFM模型 + SENET + BilinearInteraction。
    """

    def __init__(self, d_numerical, categories, d_embed,
                 deep_layers, deep_dropout, split_half=True, 
                 cross_layer_sizes=[256, 128],
                 reduction_ratio=3,
                 bilinear_type="field_interaction",
                 n_classes=1):
        
        super().__init__()
        if d_numerical is None:
            d_numerical = 0
        if categories is None:
            categories = []
            
        self.categories = categories
        self.n_classes = n_classes
        
        #! xDeepFM 是没有 num_embedding 的，它是直接把所有的值作为一个 vector
        self.num_linear = nn.Linear(d_numerical, n_classes) if d_numerical else None
        self.cat_linear = CatLinear(categories, n_classes) if categories else None
        
        #! 这边是得到 embedding
        self.num_embedding = NumEmbedding(d_numerical, 1, d_embed) if d_numerical else None
        self.cat_embedding = CatEmbedding(categories, d_embed) if categories else None
       
        #! CIN 
        num_fields = d_numerical + len(self.categories)
        self.cin = CompressedInteractionNetwork(num_fields, cross_layer_sizes, split_half)
        
        #! SENet Extraction
        self.se_attention = SENetAttention(num_fields, reduction_ratio)
        
        #! BiLinear Interaction
        self.bilinear = BilinearInteraction(num_fields, d_embed, bilinear_type)
        
        #! xDeepFM的 + FiBiNET的
        self.deep_in = d_numerical * d_embed + len(categories) * d_embed + num_fields * (num_fields - 1) * d_embed
        
        self.deep = MultiLayerPerceptron(
            d_in=self.deep_in,
            d_layers=deep_layers,
            dropout=deep_dropout,
            d_out=n_classes
        )
        

    def forward(self, x):
        
        """
        x_num: numerical features
        x_cat: category features
        """
        x_num, x_cat = x
        #!linear部分
        x = 0.0
        if self.num_linear:
            x = x + self.num_linear(x_num) 
        if self.cat_linear:
            x = x + self.cat_linear(x_cat)
        x_embedding = []
        if self.num_embedding:
            x_embedding.append(self.num_embedding(x_num[..., None]))
        if self.cat_embedding:
            x_embedding.append(self.cat_embedding(x_cat))
        x_embedding = torch.cat(x_embedding,dim=1)
        se_embedding = self.se_attention(x_embedding)
        ffm_out = self.bilinear(x_embedding)
        se_ffm_out = self.bilinear(se_embedding)
        
        #! Attention FFM 和非 Attention FFM 的 embedding 进行结合
        x_interaction = torch.flatten(torch.cat([ffm_out, se_ffm_out], dim=1), start_dim=1)
  
        #! CIN部分
        x += self.cin(x_embedding)
        
        
        flatten_x_embedding = x_embedding.view(x_interaction.shape[0], -1)
        concat_embedding = torch.cat([flatten_x_embedding, x_interaction], dim=1)
        
        #! deep部分
        x = x + self.deep(concat_embedding)
        #! 压缩成一个 scalar prob
        if self.n_classes == 1:
            x = x.squeeze(-1)
        
        return x



class MyPipeline(torch.nn.Module):
    def __init__(self, net, loss_fn, accelerator=None, metrics_dict=None, optimizer=None, lr_scheduler=None):
        super().__init__()
        self.accelerator = accelerator
        self.history = {}
        
        self.net = net
        self.loss_fn = loss_fn
        self.metrics_dict = nn.ModuleDict(metrics_dict) 
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            self.parameters(), lr=1e-2)
        self.lr_scheduler = lr_scheduler


    def forward(self, x):
        if self.net:
            return self.net.forward(x)
        else:
            raise NotImplementedError


    def fit(self, train_data=None, val_data=None, epochs=10, ckpt_path='checkpoint.pt', 
            patience=5, monitor="val_loss", mode="min"):


        for epoch in range(1, epochs+1):
            printlog("Epoch {0} / {1}".format(epoch, epochs), self.accelerator)
            
            #! 1，train -------------------------------------------------  
            train_step_runner = StepRunner(net=self.net, stage="train",
                                           loss_fn=self.loss_fn, metrics_dict=deepcopy(self.metrics_dict),
                                           optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
                                           accelerator=self.accelerator)
            train_epoch_runner = EpochRunner(train_step_runner)
            train_metrics = train_epoch_runner(train_data)
            
            for name, metric in train_metrics.items():
                #! 记录每个 epoch 的 metric
                #! 每个是一个 list 
                metric = torch.tensor(metric).float().to(self.accelerator.device)
                #! 要 gather 的话必须转成 tensor
                metric = self.accelerator.gather(metric)
                metric = metric.mean().item()
                #! 所有进程上的 history 的维护的东西都是一样的
                self.history[name] = self.history.get(name, []) + [metric]
      
          
            #! 2，validate -------------------------------------------------
            if val_data:
                val_step_runner = StepRunner(net=self.net, stage="val",
                                             loss_fn=self.loss_fn, metrics_dict=deepcopy(self.metrics_dict),
                                             accelerator=self.accelerator)
                val_epoch_runner = EpochRunner(val_step_runner)
                with torch.no_grad():
                    val_metrics = val_epoch_runner(val_data)
                val_metrics["epoch"] = epoch
                for name, metric in val_metrics.items():
                    metric = torch.tensor(metric).float().to(self.accelerator.device)
                    #! 要 gather 的话必须转成 tensor
                    metric = self.accelerator.gather(metric)
                    metric = metric.mean().item()
                    self.history[name] = self.history.get(name, []) + [metric]
                    
            
            #! 3，early-stopping -------------------------------------------------
            #! 所有的 history 维护的是一样的，所以会同时 early stop
            arr_scores = self.history[monitor]
            best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
            if best_score_idx == len(arr_scores)-1:
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_local_main_process:
                    unwrapped_model = self.accelerator.unwrap_model(self.net)
                    self.accelerator.save(unwrapped_model.state_dict(), ckpt_path)
                    self.accelerator.print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
                        arr_scores[best_score_idx]), file=sys.stderr)
                    
            if len(arr_scores) - best_score_idx > patience:
                if self.accelerator.is_local_main_process:
                    self.accelerator.print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
                        monitor, patience), file=sys.stderr)
                    #! break 之后把最好的模型 load 进来
                    #! 只在主进程上 load 模型
                    unwrapped_model.load_state_dict(torch.load(ckpt_path))
                    self.net = unwrapped_model
                #! 其他进程上的 net 是分布式的 net, 然后主进程是 unwrap 的 net
                break 
            
        return pd.DataFrame(self.history)
    
    #! 验证集
    @torch.no_grad()
    def evaluate(self, val_data):
        val_data = self.accelerator.prepare(val_data)
        val_step_runner = StepRunner(net=self.net,stage="val",
                    loss_fn=self.loss_fn, metrics_dict=deepcopy(self.metrics_dict),
                    accelerator=self.accelerator)
        val_epoch_runner = EpochRunner(val_step_runner)
        val_metrics = val_epoch_runner(val_data)
        return val_metrics
        
    #! interactive 测试
    @torch.no_grad()
    def predict(self, dataloader):
        # dataloader = self.accelerator.prepare(dataloader)
        result = torch.cat([self.forward(t[0]) for t in dataloader])
        return result.data

if __name__ == '__main__':
    
    model = DeepFM(d_numerical = 3, categories = [4,3,2],
            d_embed = 4, deep_layers = [20,20], deep_dropout=0.1,
            n_classes = 1)
    x_num = torch.randn(2,3) #! batch = 2, 3 个 number 类型特征
    x_cat = torch.randint(0,2,(2,3)) #! batch = 2, 3 个 class 特征，每个的取值都是0或1，但是每个对应的类别不止两个
    r = model((x_num, x_cat)) 
    #! [batch,]
    print(r)


