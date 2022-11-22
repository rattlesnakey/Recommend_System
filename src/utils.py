import datetime 
from tqdm import tqdm
import torch
import numpy as np
import json

def dump_json(path, item):
    with open(path, "w+") as f:
        json.dump(item, f, indent=4, sort_keys=True)


def set_seed(seed):
    np.random.seed(0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def printlog(info, accelerator):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    accelerator.print("\n"+"=========="*8 + "%s"%nowtime)
    accelerator.print(info+'...\n')
    

class StepRunner:
    def __init__(self, net, loss_fn, stage="train", metrics_dict=None, 
                 optimizer=None, lr_scheduler=None,
                 accelerator=None
                 ):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net,loss_fn, metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.accelerator = accelerator
    
    def __call__(self, features, labels):
        #!loss
        preds = self.net(features)
        loss = self.loss_fn(preds, labels)

        #!backward()
        if self.optimizer is not None and self.stage=="train":
            if self.accelerator is None:
                loss.backward()
            else:
                #! accelerator 的 backward
                self.accelerator.backward(loss)
                
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        #!metrics
        #!ModuleDict 里面取出各个指标出来计算
        step_metrics = {self.stage+"_"+name:metric_fn(preds, labels).item() 
                        for name, metric_fn in self.metrics_dict.items()}
        return loss.item(), step_metrics
    
    
class EpochRunner:
    def __init__(self, steprunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.steprunner.net.train() if self.stage=="train" else self.steprunner.net.eval()
        
    def __call__(self, dataloader):
        total_loss,step = 0, 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=self.stage+'ing ...', disable=not self.steprunner.accelerator.is_local_main_process)
        for i, batch in loop:
            features,labels = batch
            if self.stage=="train":
                loss, step_metrics = self.steprunner(features,labels)
            else:
                with torch.no_grad():
                    loss, step_metrics = self.steprunner(features,labels)
            step_log = dict({self.stage+"_loss":loss}, **step_metrics)

            total_loss += loss
            step += 1
            
            if i != len(dataloader)-1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                epoch_metrics = {self.stage+"_"+name:metric_fn.compute().item() 
                                 for name, metric_fn in self.steprunner.metrics_dict.items()}
                epoch_log = dict({self.stage+"_loss":epoch_loss}, **epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name, metric_fn in self.steprunner.metrics_dict.items():
                    metric_fn.reset()
    
        if self.steprunner.lr_scheduler is not None:
            self.steprunner.accelerator.print("当前 epoch 学习率:%f" % (self.steprunner.optimizer.param_groups[0]['lr']))
            self.steprunner.lr_scheduler.step()
        return epoch_log