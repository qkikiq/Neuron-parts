import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
# import clip
from PIL import Image
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlight import DictAction
from tools import *
# from KLLoss import KLLoss
import matplotlib.pyplot as plt

import pandas as pd
from sklearn import manifold  # 流形学习算法
import numpy as np
from scipy.special import binom
from KLLoss import KLLoss

# num_classes = 60 # ntu60
num_classes = 120 # ntu120
# unseen_classes = [10, 11, 19, 26, 56]   # ntu60_55/5_split 未见类（zero shot）
# unseen_classes = [4,19,31,47,51]   # ablation study ntu60 split1
# unseen_classes = [12,29,32,44,59]   # ablation study ntu60 split2
# unseen_classes = [7,20,28,39,58]   # ablation study ntu60 split3
# unseen_classes = [3,5,9,12,15,40,42,47,51,56,58,59]  # ntu60_48/12_split
unseen_classes = [4,13,37,43,49,65,88,95,99,106]  # ntu120_110/10_split
# unseen_classes = [5,9,11,16,18,20,22,29,35,39,45,49,59,68,70,81,84,87,93,94,104,113,114,119]  # ntu120_96/24_split

# unseen_classes = [1, 9, 20, 34, 50]  # pkuv1 46/5 split
# unseen_classes = [3,7,11,15,19,21,25,31,33,36,43,48]   # pkuv1 39/12 split
# unseen_classes = [3,14,29,31,49]  # pkuv1 46/5 ablation study split1
# unseen_classes = [2,15,39,41,43]  # pkuv1 46/5 ablation study split2
# unseen_classes = [4,12,16,22,36]  # pkuv1 46/5 ablation study split3

#除去unseen_classes后的类别即为seen_classes
seen_classes = list(set(range(num_classes))-set(unseen_classes))  # ntu120
train_label_dict = {}  # 训练集标签字典->one-hot向量
train_label_map_dict = {} # 训练集标签映射字典->索引
# 遍历已见类别，构建 one-hot 编码
for idx, l in enumerate(seen_classes):
    tmp = [0] * len(seen_classes)  # 创建长度为已见类别数的零向量
    tmp[idx] = 1
    train_label_dict[l] = tmp  # 类别ID -> one-hot向量(单位矩阵)
    train_label_map_dict[l] = idx # 类别ID -> 索引（除去unseen重新排序）
test_zsl_label_dict = {}  # 测试集标签字典->one-hot向量
test_zsl_label_map_dict = {} # 测试集标签映射字典->索引
for idx, l in enumerate(unseen_classes):
    tmp = [0] * len(unseen_classes)
    tmp[idx] = 1
    test_zsl_label_dict[l] = tmp
    test_zsl_label_map_dict[l] = idx
test_gzsl_label_dict = {} # GZSL测试集标签字典->one-hot向量（单位向量）
test_gzsl_label_map_dict = {} # GZSL测试集标签映射字典->索引（映射为新列表）
for idx, l in enumerate(range(num_classes)):
    tmp = [0] * num_classes
    tmp[idx] = 1
    test_gzsl_label_dict[l] = tmp
    test_gzsl_label_map_dict[l] = idx


scaler = torch.cuda.amp.GradScaler() # 自动混合精度缩放器（提升训练效率）

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0

        # 载入空间/时间语义特征（预先计算/缓存的文本/阶段描述张量） 维度是（120，10，768）
        self.spatial_round_fg = torch.load('/dadaY/xinyu/pycharmproject/Neuron/semantics/ntu/spatial/ntu120_gpt4omini_incontext_spatial_fg.tar')
        self.spatial_round_mg = torch.load('/dadaY/xinyu/pycharmproject/Neuron/semantics/ntu/spatial/ntu120_gpt4omini_incontext_spatial_mg.tar')
        self.spatial_round_cg = torch.load('/dadaY/xinyu/pycharmproject/Neuron/semantics/ntu/spatial/ntu120_gpt4omini_incontext_spatial_cg.tar')
        self.temporal_round_cg = torch.load('/dadaY/xinyu/pycharmproject/Neuron/semantics/ntu/temporal/ntu120_gpt4omini_incontext_temporal_fp.tar')
        self.temporal_round_mg = torch.load('/dadaY/xinyu/pycharmproject/Neuron/semantics/ntu/temporal/ntu120_gpt4omini_incontext_temporal_fsp.tar')
        self.temporal_round_fg = torch.load('/dadaY/xinyu/pycharmproject/Neuron/semantics/ntu/temporal/ntu120_gpt4omini_incontext_temporal_fstp.tar')
        # self.spatial_round_fg = torch.load('/root/autodl-tmp/FGE/semantics/pkuv1/spatial/pkuv1_gpt4omini_incontext_spatial_fg.tar')
        # self.spatial_round_mg = torch.load('/root/autodl-tmp/FGE/semantics/pkuv1/spatial/pkuv1_gpt4omini_incontext_spatial_mg.tar')
        # self.spatial_round_cg = torch.load('/root/autodl-tmp/FGE/semantics/pkuv1/spatial/pkuv1_gpt4omini_incontext_spatial_cg.tar')
        # self.temporal_round_cg = torch.load('/root/autodl-tmp/FGE/semantics/pkuv1/temporal/pkuv1_gpt4omini_incontext_temporal_fp.tar')
        # self.temporal_round_mg = torch.load('/root/autodl-tmp/FGE/semantics/pkuv1/temporal/pkuv1_gpt4omini_incontext_temporal_fsp.tar')
        # self.temporal_round_fg = torch.load('/root/autodl-tmp/FGE/semantics/pkuv1/temporal/pkuv1_gpt4omini_incontext_temporal_fstp.tar')
        print('Extract CLIP Semantics Successful!')

        # #todo
        # data = np.load("view/ntu120_body_parts_embeddings.npz")
        # # 获取左臂的嵌入
        # self.left_arm = data['left_arm']
        # # 获取右臂的嵌入
        # self.right_arm = data['right_arm']
        # # 获取左腿的嵌入
        # self.left_leg = data['left_leg']
        # # 获取右腿的嵌入
        # self.right_leg = data['right_leg']
        # # 获取头部和躯干的嵌入
        # self.h_torso = data['h_torso']
        # # 获取下半身躯干的嵌入
        # self.w_torso = data['w_torso']
        # # 合并所有身体部位的嵌入
        # self.all_part = np.concatenate((self.left_arm, self.right_arm, self.left_leg, self.right_leg, self.h_torso, self.w_torso), axis=1)

        # load model
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.model = self.model.cuda(self.output_device)

        # load skeleton action recognition model
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)
        

        
    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def load_model(self):
        # 获取输出设备（GPU编号），如果是列表则取第一个，否则直接用
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        # 动态导入模型类
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model) # 打印模型类信息
        self.model = Model(**self.arg.model_args) # 打印模型类信息
        # print(self.model)
        # loss function setting
        self.loss_ce = nn.CrossEntropyLoss().cuda(output_device) # 设置交叉熵损失函数并放到GPU
        self.loss_kl = KLLoss().cuda(output_device) # 设置KL损失函数并放到GPU

        # 如果指定了权重文件，则加载权重
        if self.arg.weights:
            self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            # 根据文件后缀选择加载方式
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            # 处理权重字典，适配feature_extractor和多卡训练
            weights = OrderedDict([["feature_extractor."+k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
            # 遍历需要忽略的权重名，移除对应权重
            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))
            # 加载权重到模型
            try:
                self.model.load_state_dict(weights)
            except:
                # 如果权重不完全匹配，打印缺失的权重名
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                    # 用已有权重更新模型参数
                state.update(weights)
                self.model.load_state_dict(state)
        print("Load model done.")

        
    def load_optimizer(self):
        # 判断优化器类型是否为SGD
        if self.arg.optimizer == 'SGD':
            # 使用SGD优化器，只优化需要梯度的参数，设置学习率、动量、Nesterov动量和权重衰减
            self.optimizer = optim.SGD(
                filter(lambda p:p.requires_grad, self.model.parameters()),
                lr=self.arg.base_lr,
                momentum=0.9, 
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay) #权重衰减
        else:
            raise ValueError()
        
    
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),  # 构建训练集数据集对象
                batch_size=self.arg.batch_size,  # 训练批次大小
                shuffle=True,  # 打乱数据
                num_workers=self.arg.num_worker,  # 加载数据的线程数
                drop_last=True,  # 丢弃最后不足一个批次的数据
                worker_init_fn=init_seed)  # 初始化每个worker的随机种子
            print("Load train data done.")  # 打印训练集加载完成
            # 加载零样本测试集
        self.data_loader['test_zsl'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_zsl_args),  # 构建ZSL测试集数据集对象
            batch_size=self.arg.test_batch_size,  # 测试批次大小
            shuffle=False,  # 不打乱数据
            num_workers=self.arg.num_worker,  # 加载数据的线程数
            drop_last=False,  # 不丢弃最后不足一个批次的数据
            worker_init_fn=init_seed)  # 初始化每个worker的随机种子
        print("Load zsl test data done.")  # 打印ZSL测试集加载完成
        # 加载广义零样本测试集
        self.data_loader['test_gzsl'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_gzsl_args),  # 构建GZSL测试集数据集对象
            batch_size=self.arg.test_batch_size,  # 测试批次大小
            shuffle=False,  # 不打乱数据
            num_workers=self.arg.num_worker,  # 加载数据的线程数
            drop_last=False,  # 不丢弃最后不足一个批次的数据
            worker_init_fn=init_seed)  # 初始化每个worker的随机种子
        print("Load gzsl test data done.")  # 打印GZSL测试集加载完成

    
    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            # 如果当前轮次小于预热轮次，则线性增加学习率
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                # 否则按设定的衰减步长和衰减率调整学习率
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            #更新优化器中每个参数组的学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()
    
    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time
    
    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train() #训练模式
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train'] #训练数据加载器
        self.adjust_learning_rate(epoch)    #学习率调整

        loss_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001) # 计时器
        process = tqdm(loader, ncols=40) # 进度条

        # semantics
        # 生成已见类的空间/时间语义张量（切片后上 GPU）
        spatial_round_fg_seen = self.spatial_round_fg[seen_classes].cuda(self.output_device)
        spatial_round_mg_seen = self.spatial_round_mg[seen_classes].cuda(self.output_device)
        spatial_round_cg_seen = self.spatial_round_cg[seen_classes].cuda(self.output_device)
        spatial_round_seen = [spatial_round_cg_seen, spatial_round_mg_seen, spatial_round_fg_seen]
        temporal_round_cg_seen = self.temporal_round_cg[seen_classes].cuda(self.output_device)
        temporal_round_mg_seen = self.temporal_round_mg[seen_classes].cuda(self.output_device)
        temporal_round_fg_seen = self.temporal_round_fg[seen_classes].cuda(self.output_device)
        temporal_round_seen = [temporal_round_cg_seen, temporal_round_mg_seen, temporal_round_fg_seen]



        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                b,_,_,_,_ = data.size()
            timer['dataloader'] += self.split_time()
            self.optimizer.zero_grad()

            label_reindex_seen = torch.tensor([train_label_map_dict[l.item()] for l in label]).cuda(self.output_device)
            output_spa, output_tem = self.model(data, spatial_round_seen, temporal_round_seen)
            loss_spa = (self.loss_ce(output_spa[0], label_reindex_seen)+self.loss_ce(output_spa[1], label_reindex_seen)+self.loss_ce(output_spa[2], label_reindex_seen))/3
            loss_tem = (self.loss_ce(output_tem[0], label_reindex_seen)+self.loss_ce(output_tem[1], label_reindex_seen)+self.loss_ce(output_tem[2], label_reindex_seen))/3
            loss = (loss_spa + loss_tem)/2
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()
            # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))



        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')


    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        # semantics
        # zsl
        spatial_round_fg_unseen = self.spatial_round_fg[unseen_classes].cuda(self.output_device)
        spatial_round_mg_unseen = self.spatial_round_mg[unseen_classes].cuda(self.output_device)
        spatial_round_cg_unseen = self.spatial_round_cg[unseen_classes].cuda(self.output_device)
        spatial_round = [spatial_round_cg_unseen, spatial_round_mg_unseen, spatial_round_fg_unseen]
        temporal_round_cg_unseen = self.temporal_round_cg[unseen_classes].cuda(self.output_device)
        temporal_round_mg_unseen = self.temporal_round_mg[unseen_classes].cuda(self.output_device)
        temporal_round_fg_unseen = self.temporal_round_fg[unseen_classes].cuda(self.output_device)
        temporal_round = [temporal_round_cg_unseen,temporal_round_mg_unseen,temporal_round_fg_unseen]
        # gzsl
        gzsl_spatial_round_fg_unseen = self.spatial_round_fg[:num_classes].cuda(self.output_device)
        gzsl_spatial_round_mg_unseen = self.spatial_round_mg[:num_classes].cuda(self.output_device)
        gzsl_spatial_round_cg_unseen = self.spatial_round_cg[:num_classes].cuda(self.output_device)
        gzsl_spatial_round = [gzsl_spatial_round_cg_unseen, gzsl_spatial_round_mg_unseen, gzsl_spatial_round_fg_unseen]
        gzsl_temporal_round_cg_unseen = self.temporal_round_cg[:num_classes].cuda(self.output_device)
        gzsl_temporal_round_mg_unseen = self.temporal_round_mg[:num_classes].cuda(self.output_device)
        gzsl_temporal_round_fg_unseen = self.temporal_round_fg[:num_classes].cuda(self.output_device)
        gzsl_temporal_round = [gzsl_temporal_round_cg_unseen,gzsl_temporal_round_mg_unseen,gzsl_temporal_round_fg_unseen]

        for ln in loader_name:
            spa_pred_list = [[],[],[]]
            spa_pred_list_fg = []
            spa_pred_list_mg = []
            spa_pred_list_cg = []
            tem_pred_list = [[],[],[]]
            tem_pred_list_fg = []
            tem_pred_list_mg = []
            tem_pred_list_cg = []
            # cat_pred_list = []
            spa_feat_list = [[],[],[]]
            tem_feat_list = [[],[],[]]
            cube_feat_list = []
            ta_feat_list = [[],[],[]]
            label_list = []
            loss_value = []
            score_frag = []
            sim_matrix_list = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)
            sp_norm_feature_list = []
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    b, _, _, _, _ = data.size()
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    # 零样本测试集
                    if ln == "test_zsl":
                        # 标签重新索引为未见类索引
                        label_reindex_unseen = torch.tensor([test_zsl_label_map_dict[l.item()] for l in label]).cuda(self.output_device)
                        # 前向推理，获得空间和时间分支输出
                        output_spa, output_tem = self.model(data, spatial_round, temporal_round)
                        # 统计每个分支的预测类别
                        for i in range(3):
                            spa_pred_list[i].append(torch.max(output_spa[i].data, 1)[1].data.cpu().numpy())
                            tem_pred_list[i].append(torch.max(output_tem[i].data, 1)[1].data.cpu().numpy())
                       #保存标签
                        label_list.append(label_reindex_unseen.data.cpu().numpy())

                    if ln == "test_gzsl":
                        # 标签重新索引为所有类索引
                        label_reindex_unseen = torch.tensor([test_gzsl_label_map_dict[l.item()] for l in label]).cuda(self.output_device)
                        # 前向推理，获得空间和时间分支输出
                        output_spa, output_tem = self.model(data, gzsl_spatial_round, gzsl_temporal_round)
                        # 对分支输出做softmax
                        spa_predict_label_fg = F.softmax(output_spa[2], 1)
                        tem_predict_label_fg = F.softmax(output_tem[2], 1)
                        # 保存分支预测分数
                        spa_pred_list_fg.append(spa_predict_label_fg)
                        tem_pred_list_fg.append(tem_predict_label_fg)
                        # 保存标签
                        label_list.append(label_reindex_unseen.data.cpu().numpy())
                    step += 1
            # 统计ZSL准确率
            if ln == 'test_zsl':
                label_list= np.concatenate(label_list)
                for i in range(3):
                    spa_pred_list[i] = np.concatenate(spa_pred_list[i])  # 合并空间分支预测
                    tem_pred_list[i] = np.concatenate(tem_pred_list[i]) # 合并时间分支预测
                spa_acc_fg = np.mean((spa_pred_list[2]==label_list)) # 空间分支准确率
                tem_acc_fg = np.mean((tem_pred_list[2]==label_list)) # 时间分支准确率
                pred_sum_last = ((spa_pred_list[2]==label_list)*1+(tem_pred_list[2]==label_list)*1)  # 时间分支准确率
                pred_sum_last[pred_sum_last > 0] = 1 # 只要有一个分支预测正确就算对
                acc_or_last = np.mean(pred_sum_last) # 总准确率
                print('*'*100)
                self.print_log('\tTop{} Acc: {:.2f}%'.format(1, acc_or_last*100))
                

            if ln == 'test_gzsl': # 如果当前评估的是GZSL（广义零样本学习）测试集
                label_list= np.concatenate(label_list) # 合并所有批次的标签为一个数组
                sim_matrix_spa = torch.cat(spa_pred_list_fg,dim=0).cuda(self.output_device) # 合并所有批次的标签为一个数组
                sim_matrix_tem = torch.cat(tem_pred_list_fg,dim=0).cuda(self.output_device) # 拼接所有时间分支的预测分数，并转到GPU
                calibration_factor_spa_list = [i/100000 for i in range(20, 31, 1)] # 空间分支校准因子列表（用于置信度调整）
                calibration_factor_tem_list = [i/100000 for i in range(20, 31, 1)] # 时间分支校准因子列表
                result_spa = []  # 存储空间分支的评估结果
                result_tem = []  # 存储时间分支的评估结果
                result = []  # 存储融合分支的评估结果

                # 遍历所有校准因子组合
                for cf_spa in calibration_factor_spa_list:
                    # spa 校准空间分支分数
                    sim_matrix_spa_loop = sim_matrix_spa.clone()  # 复制空间分支分数矩阵
                    tmp = torch.zeros_like(sim_matrix_spa_loop)  # 创建同形状的零矩阵
                    tmp[:, seen_classes] = cf_spa   # 已见类赋值为校准因子
                    sim_matrix_spa_loop = sim_matrix_spa_loop - tmp  # 对已见类置信度减去校准因子
                    sim_matrix_pred_spa_idx = torch.max(sim_matrix_spa_loop, dim=1)[1] # 获取每个样本预测的类别索引
                    sim_matrix_pred_spa_idx = sim_matrix_pred_spa_idx.data.cpu().numpy() # 转为numpy数组
                    acc_spa_seen = []# 已见类准确率列表
                    acc_spa_unseen = []  # 未见类准确率列表
                    #真实标签tl   预测标签pl
                    for tl, pl in zip(label_list, sim_matrix_pred_spa_idx):
                        if tl in seen_classes:
                            acc_spa_seen.append(int(tl)==int(pl))
                        else:
                            acc_spa_unseen.append(int(tl)==int(pl))
                    acc_spa_seen = sum(acc_spa_seen) / len(acc_spa_seen)  # 已见类平均准确率
                    acc_spa_unseen = sum(acc_spa_unseen) / len(acc_spa_unseen)  # 未见类平均准确率
                    harmonic_mean_acc_spa = 2*acc_spa_seen*acc_spa_unseen/(acc_spa_seen+acc_spa_unseen)
                    result_spa.append((cf_spa, acc_spa_unseen, acc_spa_seen, harmonic_mean_acc_spa))
                    # tem 校准空间分支分数
                    for cf_tem in calibration_factor_tem_list:
                        sim_matrix_tem_loop = sim_matrix_tem.clone()
                        tmp = torch.zeros_like(sim_matrix_tem_loop)
                        tmp[:, seen_classes] = cf_tem
                        sim_matrix_tem_loop = sim_matrix_tem_loop - tmp
                        sim_matrix_pred_tem_idx = torch.max(sim_matrix_tem_loop, dim=1)[1]
                        sim_matrix_pred_tem_idx = sim_matrix_pred_tem_idx.data.cpu().numpy()
                        acc_tem_seen = []
                        acc_tem_unseen = []
                        for tl, pl in zip(label_list, sim_matrix_pred_tem_idx):
                            if tl in seen_classes:
                                acc_tem_seen.append(int(tl)==int(pl))
                            else:
                                acc_tem_unseen.append(int(tl)==int(pl))
                        acc_tem_seen = sum(acc_tem_seen) / len(acc_tem_seen)
                        acc_tem_unseen = sum(acc_tem_unseen) / len(acc_tem_unseen)
                        harmonic_mean_acc_tem = 2*acc_tem_seen*acc_tem_unseen/(acc_tem_seen+acc_tem_unseen)
                        result_tem.append((cf_tem, acc_tem_unseen, acc_tem_seen, harmonic_mean_acc_tem))

                        # add
                        # 统计空间分支和时间分支预测正确的样本（只要有一个分支预测正确就算对）
                        pred_sum_last = ((sim_matrix_pred_spa_idx==label_list)*1+(sim_matrix_pred_tem_idx==label_list)*1)
                        # 将预测正确的样本标记为1（有分支预测正确即为1）
                        pred_sum_last[pred_sum_last > 0] = 1
                        acc_seen = [] # 已见类准确率列表
                        acc_unseen = [] # 未见类准确率列表
                        # 遍历每个样本的真实标签和融合分支预测结果
                        for tl, pl in zip(label_list, pred_sum_last):
                            if tl in seen_classes:
                                acc_seen.append(pl)
                            else:
                                acc_unseen.append(pl)
                        acc_seen = sum(acc_seen) / len(acc_seen)
                        acc_unseen = sum(acc_unseen) / len(acc_unseen)
                        harmonic_mean_acc = 2*acc_seen*acc_unseen/(acc_seen+acc_unseen)
                        result.append((cf_spa, cf_tem, acc_unseen, acc_seen,acc_spa_seen, acc_spa_unseen,acc_tem_seen, acc_tem_unseen, harmonic_mean_acc, harmonic_mean_acc_spa, harmonic_mean_acc_tem))

                print('*'*100)
                best_calibration_spa_factor = -1 # 初始化最佳空间校准因子
                best_calibration_tem_factor = -1  # 初始化最佳时间校准因子
                best_accuracy_unseen = -1    # 初始化最佳未见类准确率
                best_accuracy_seen = -1   # 初始化最佳已见类准确率
                best_accuracy_spa_unseen = -1 # 初始化空间分支最佳未见类准确率
                best_accuracy_spa_seen = -1 # 初始化空间分支最佳已见类准确率
                best_accuracy_tem_unseen = -1 # 初始化时间分支最佳未见类准确率
                best_accuracy_tem_seen = -1 # 初始化时间分支最佳已见类准确率
                best_harmonic_mean_acc = -1 # 初始化最佳调和平均准确率
                best_harmonic_mean_acc_spa = -1 # 初始化空间分支最佳调和平均准确率
                best_harmonic_mean_acc_tem = -1 # 初始化时间分支最佳调和平均准确率
                # 遍历所有校准因子组合下的评估结果
                for cf_spa, cf_tem, accuracy_unseen, accuracy_seen,acc_spa_seen, acc_spa_unseen,acc_tem_seen, acc_tem_unseen,harmonic_mean_acc, harmonic_mean_acc_spa,harmonic_mean_acc_tem in result:
                    # 如果当前融合分支调和平均准确率更高，则更新最佳结果
                    if harmonic_mean_acc > best_harmonic_mean_acc:
                        self.best_acc_epoch = epoch + 1
                        best_harmonic_mean_acc = harmonic_mean_acc
                        best_harmonic_mean_acc_spa = harmonic_mean_acc_spa
                        best_harmonic_mean_acc_tem = harmonic_mean_acc_tem
                        best_accuracy_unseen = accuracy_unseen
                        best_accuracy_seen = accuracy_seen
                        best_accuracy_spa_unseen = acc_spa_unseen
                        best_accuracy_spa_seen = acc_spa_seen
                        best_accuracy_tem_unseen = acc_tem_unseen
                        best_accuracy_tem_seen = acc_tem_seen
                        best_calibration_spa_factor = cf_spa
                        best_calibration_tem_factor = cf_tem # 更新最佳时间校准因子
                self.print_log('\tCalibration Spatial Factor: {:.8f}'.format(best_calibration_spa_factor)) # 打印空间校准因子（用于调整已见类置信度）
                self.print_log('\tCalibration Temporal Factor: {:.8f}'.format(best_calibration_tem_factor)) # 打印时间校准因子（用于调整已见类置信度）
                self.print_log('\tSeen Spa Acc: {:.2f}%'.format(best_accuracy_spa_seen*100)) # 打印空间分支已见类的准确率
                self.print_log('\tUnseen Spa Acc: {:.2f}%'.format(best_accuracy_spa_unseen*100)) # 打印空间分支未见类的准确率
                self.print_log('\tHarmonic Mean Spa Acc: {:.2f}%'.format(best_harmonic_mean_acc_spa*100)) # 打印空间分支已见/未见类的调和平均准确率
                self.print_log('\tSeen Tem Acc: {:.2f}%'.format(best_accuracy_tem_seen*100)) # 打印时间分支已见类的准确率
                self.print_log('\tUnseen Tem Acc: {:.2f}%'.format(best_accuracy_tem_unseen*100)) # 打印时间分支未见类的准确率
                self.print_log('\tHarmonic Mean Tem Acc: {:.2f}%'.format(best_harmonic_mean_acc_tem*100)) # 打印时间分支已见/未见类的调和平均准确率
                self.print_log('\tSeen Acc: {:.2f}%'.format(best_accuracy_seen*100)) # 打印融合分支已见类的准确率
                self.print_log('\tUnseen Acc: {:.2f}%'.format(best_accuracy_unseen*100)) # 打印融合分支未见类的准确率
                self.print_log('\tHarmonic Mean Acc: {:.2f}%'.format(best_harmonic_mean_acc*100)) # 打印融合分支未见类的准确率

    def start(self):
        if self.arg.phase == 'train':  # 如果是训练阶段
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model): # 统计模型参数数量
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            start_epoch = 0
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):  # 遍历每个epoch
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                self.train(epoch, save_model=True)
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test_zsl']) # ZSL测试
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test_gzsl']) # GZSL测试
                
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0] # 获取最佳模型权重路径
            weights = torch.load(weights_path) # 加载权重
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()]) # 权重适配多卡
            self.model.load_state_dict(weights) # 加载权重到模型

            wf = weights_path.replace('.pt', '_wrong.txt') # 错误样本文件路径
            rf = weights_path.replace('.pt', '_right.txt')# 正确样本文件路径
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf) # 测试并保存结果

            self.eval(epoch=0, save_score=True, loader_name=['test_zsl'], wrong_file=wf, result_file=rf)  # ZSL测试
            self.eval(epoch=0, save_score=True, loader_name=['test_gzsl'], wrong_file=wf, result_file=rf)  # GZSL测试
            self.arg.print_log = True


            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) # 测试并保存结果
            self.print_log(f'Best accuracy: {self.best_acc}') # 打印最佳准确率
            self.print_log(f'Epoch number: {self.best_acc_epoch}') # 打印最佳epoch
            self.print_log(f'Model name: {self.arg.work_dir}') # 打印最佳epoch
            self.print_log(f'Model total number of params: {num_params}') # 打印参数总数
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt') #错误样本记录文件
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.') # 测试阶段必须提供权重
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights)) # 打印权重文件路径
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)  #评估
            self.print_log('Done.\n')
    

def gen_label(labels):
    num = len(labels)  # 样本数
    gt = numpy.zeros(shape=(num,num)) # 初始化一个num x num的零矩阵
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label: # 若第 i 与第 k 样本同类，置 1
                gt[i,k] = 1
    return gt


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    # 将传入的字符串按最后一个点分割，得到模块名和类名
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    parser = argparse.ArgumentParser(description='LLMs for Action Recognition') # 创建参数解析器，描述为动作识别的LLMs
    parser.add_argument('--work-dir', default='./work_dir/temp', help='the work folder for stroing results.') #保存目录
    parser.add_argument('-model_saved_name', default='') #模型保存名称
    parser.add_argument('--config', default='./config/nturgbd-cross-view/default.yaml', help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test') #阶段，训练或测试
    parser.add_argument('--save-score', type=str2bool, default=False, help='if ture, the classification score will be stored') #是否保存分类分数

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch') #随机种子
    parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)') #日志打印间隔
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)') #模型保存间隔
    parser.add_argument('--save-epoch', type=int, default=30, help='the start epoch to save model (#iteration)')    #开始保存模型的epoch
    parser.add_argument('--eval-interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-topk', type=int, default=[1], nargs='+', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used') #数据加载器类名
    parser.add_argument('--num-worker', type=int, default=32, help='the number of worker for data loader') #线程数
    parser.add_argument('--train-feeder-args', action=DictAction, default=dict(), help='the arguments of data loader for training') #训练数据加载器参数
    parser.add_argument('--test-feeder-zsl-args', action=DictAction, default=dict(), help='the arguments of data loader for test zsl') #测试ZSL数据加载器参数
    parser.add_argument('--test-feeder-gzsl-args', action=DictAction, default=dict(), help='the arguments of data loader for test gzsl') #测试GZSL数据加载器参数

    # model
    parser.add_argument('--model', default=None, help='the model will be used') #模型类名
    parser.add_argument('--model-args', action=DictAction, default=dict(), help='the arguments of model') #模型参数
    parser.add_argument('--weights', default=None, help='the weights for network initialization') #模型权重文件路径
    parser.add_argument('--text_weights', default=None, help='the weights for network initialization') #文本权重文件
    parser.add_argument('--rgb_weights', default=None, help='the weights for network initialization') #RGB权重文件
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization') #初始化时忽略的权重名

    # optim
    parser.add_argument('--base-lr', type=float, default=0.001, help='initial learning rate') #初始学习率
    parser.add_argument('--step', type=int, default=[20, 40, 60], nargs='+', help='the epoch where optimizer reduce the learning rate') #学习率衰减的epoch
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing') #使用的GPU设备
    parser.add_argument('--optimizer', default='Adam', help='type of optimizer') #优化器类型
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not') #是否使用Nesterov动量
    parser.add_argument('--batch-size', type=int, default=256, help='training batch size') #训练批次大小
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size') #测试批次大小
    parser.add_argument('--start-epoch', type=int, default=0, help='start training from which epoch') #从哪个epoch开始训练
    parser.add_argument('--num-epoch', type=int, default=80, help='stop training in which epoch')   #训练总迭代次数
    parser.add_argument('--weight-decay', type=float, default=0.01, help='weight decay for optimizer') #权重衰减系数
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate') #学习率衰减率
    parser.add_argument('--warm_up_epoch', type=int, default=0) #预热epoch数
    parser.add_argument('--loss-alpha1', type=float, default=0.8) #损失函数权重系数1
    parser.add_argument('--loss-alpha2', type=float, default=0.8)
    parser.add_argument('--loss-alpha3', type=float, default=0.8)
    parser.add_argument('--te-lr-ratio', type=float, default=1)

    return parser


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()

