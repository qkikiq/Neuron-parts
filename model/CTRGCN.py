import math
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        cube_feature = x.view(N, M, c_new, -1, V)  # N, M, C, T, V
        cube_feature = cube_feature.mean(1) # N, C, T, V
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x), x, cube_feature

# class ModelMatch(nn.Module):
#     def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
#         super(ModelMatch, self).__init__()
# # 预训练的骨架特征提取器（返回分类输出、全局池化特征、时空立方体特征）
#         self.feature_extractor = Model(num_class, num_point, num_person, graph, graph_args, in_channels)
#         for p in self.parameters():
#             p.requires_grad = False  # 冻结当前模块所有参数（包含 feature_extractor），只训练下方原型相关模块
#
#     def forward(self, x, spatial_round, temporal_round):
#         #通过特征提取器: _为分类输出 (未使用)，pooling_feature为全局池化特征，cube_feature为时空立方体特征
#         # cube_feature 形状假设为 (B, 256, T=16, V=25)，与注释中常见设定一致
#
#         #分类（N，120）   全局池化(N C)  时空立方体（N C T V）
#         _, pooling_feature, cube_feature = self.feature_extractor(x)  # n, 256    n, 256, 16, 25
#         b, _, _, _ = cube_feature.size()


class ModelMatch(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(ModelMatch, self).__init__()
        # pretraining model
        # 预训练的骨架特征提取器（返回分类输出、全局池化特征、时空立方体特征）
        self.feature_extractor = Model(num_class, num_point, num_person, graph, graph_args, in_channels)
        for p in self.parameters():
            p.requires_grad = False  # 冻结当前模块所有参数（包含 feature_extractor），只训练下方原型相关模块
        # 原型字典：80 个“空间/时间”原型，每个维度 256
        self.spatial_prototype = nn.Embedding(80, 256)  # 空间原型 256 25*4
        self.temporal_prototype = nn.Embedding(80, 256)  # 时间原型 256 16*3
        self.relu = nn.ReLU()

        # 空间方向上的更新块（两层 MLP），用于对原型/相关聚合后的特征进行非线性变换
        self.fc_spatial = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        # 将 256 维空间表征映射到 768 维（与语义向量维度对齐，便于后续相似度计算）
        self.spatial_project = nn.Sequential(
            nn.Linear(256, 768),
            nn.ReLU()
        )
        # 三轮迭代更新的空间 MLP（深拷贝 3 份，逐轮使用）
        self.update_spatial = nn.ModuleList([copy.deepcopy(self.fc_spatial) for i in range(3)])

        # 时间方向上的更新块（结构同上）
        self.fc_temporal = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        # 将 256 维时间表征映射到 768 维（与语义对齐）
        self.temporal_project = nn.Sequential(
            nn.Linear(256, 768),
            nn.ReLU()
        )
        # 三轮迭代更新的时间 MLP
        self.update_temporal = nn.ModuleList([copy.deepcopy(self.fc_temporal) for i in range(3)])

        # 记忆门与回忆门（使用 fc_spatial 结构）用于时间原型的门控更新（类似 GRU 的 remember/recall）
        self.memory_temporal = nn.ModuleList([copy.deepcopy(self.fc_spatial) for i in range(3)])
        self.recall_temporal = nn.ModuleList([copy.deepcopy(self.fc_spatial) for i in range(3)])

                    #data  空间语义 时间语义
    def forward(self, x, spatial_round, temporal_round):
        #通过特征提取器: _为分类输出 (未使用)，pooling_feature为全局池化特征，cube_feature为时空立方体特征
        # cube_feature 形状假设为 (B, 256, T=16, V=25)，与注释中常见设定一致

        #分类（N，120）   全局池化(N C)  时空立方体（N C T V）
        _, pooling_feature, cube_feature = self.feature_extractor(x)  # n, 256    n, 256, 16, 25
        b, _, _, _ = cube_feature.size()

        #==================== 空间原型匹配与更新 =================#
        # prototype
        #取出空间原型权重：(80,256) -> 扩展到 batch 后 (B,80,256) -> 转置为 (B,256,80)，便于与 (B,256,25) 做相关
        sp = self.spatial_prototype.weight.unsqueeze(0).expand(b, -1, -1).permute(0,2,1)
        sp_list = [] # 存放每一轮迭代得到的样本级空间向量 (B,256)

        for idx, dec in enumerate(self.update_spatial):
            # 计算空间相关性：cube_feature 在时间维度上均值池化，得到 (B,256,25)
            # 选择分数：对每个关节 j，与每个原型 v 做相似度；einsum 'ncj,ncv->njv'
            # sp_select 形状 (B, 25, 80)：每个关节对每个原型的打分
            sp_select = torch.einsum('ncj,ncv->njv', cube_feature.mean(2), sp)  # n, 25, 100(a)
            # 对原型维做 softmax，得到“每个关节在 80 个原型上的注意力”
            correlation_value_spatial = F.softmax(sp_select, dim=2)  # n, 25, 100(a)
            _, _, att_num = correlation_value_spatial.size()

            # 选取当前轮要“屏蔽/弱化”的原型索引：取 topk(80) 并保留末尾 20、40、60（逐轮增加）个弱原型
            pro_indices = correlation_value_spatial.topk(80, dim=2)[1][:,:,80-(idx+1)*20:]  # 20
            # 构造 batch 与关节的索引，用于掩码操作
            batch_indices = torch.arange(b).unsqueeze(1).unsqueeze(2)
            joint_indices = torch.arange(25).unsqueeze(0).unsqueeze(2)
            # 再对关节维做 softmax，得到“每个原型在 25 个关节上的注意力”
            corr = F.softmax(sp_select, dim=1)  # (B,25,80)

            # 构建掩码：默认 1，选中的原型位置赋极小值，抑制其权重（实现迭代挑选不同原型）
            mask = torch.ones((b,25,80)).cuda()
            mask[batch_indices,joint_indices,pro_indices] = 1e-12
            corr = corr * mask #
            # 按关节聚合到原型
            sp = torch.einsum('ncj,njv->nvc', cube_feature.mean(2), corr) # n 75(a)  256
            sp = dec(sp).permute(0,2,1)  # n 256 75(a)
            sp_list.append(sp.mean(2))

            #将三轮得到的 (B,256) 投影到 768，并做 L2 归一化，便于与语义向量计算余弦相似度
        sp_proj_list = []
        for sp_ele in sp_list:
            sp_proj_list.append(F.normalize(self.spatial_project(sp_ele), p=2, dim=1))

        # prototype
        # ===== 时间原型匹配部分 =====
        # 取出时间原型权重：(80,256) -> (B,256,80)
        tp = self.temporal_prototype.weight.unsqueeze(0).expand(b, -1, -1).permute(0,2,1)

        # 时间特征：对关节维做平均（仅保留时间变化），(B,256,T)
        temporal_feature = cube_feature.mean(3)
        crop_feature = temporal_feature # 初始为完整时间长度
        # 分三种时间尺度进行匹配 ：前5帧 前10帧 全部帧
        crop_feature_list = [temporal_feature[:,:,:5], temporal_feature[:,:,:10], temporal_feature]
        tp_list = [] # 存三轮的时间样本级表征 (B,256)
        tp_attention_list = []   # 可存注意力分数（未返回）
        for idx, dec in enumerate(self.update_temporal):
            crop_feature = crop_feature_list[idx] # 选当前尺度的时间片 (B,256,t)
            # 计算每一帧与每个原型的相似度：'nct,nca->nta'，得到 (B,t,80)
            tp_select = torch.einsum('nct,nca->nta', crop_feature, tp)  # n frame_num 64(a)
            # 对帧维做 softmax：得到“每个原型在各帧上的注意力分布”（或反之，视解释而定）
            correlation_value_temporal = F.softmax(tp_select, dim=1)  # n, frame_num, 64(a)
            _, frame_num, att_num = correlation_value_temporal.size()
            tp_attention_list.append(tp_select)

            #以注意力加权聚合到原型维
            tp_cur = torch.einsum('nct,nta->nac', crop_feature, correlation_value_temporal) # n frame_num  256

            # 门控更新：回忆门 recall 与记忆门 remember
            # recall：对 tp_cur 通过 MLP 后 sigmoid，再与旧原型 tp 相乘（像门控把旧知识取回）
            tp_recall = F.sigmoid(self.recall_temporal[idx](tp_cur).permute(0,2,1)) * tp
            # remember：对 tp_cur 通过 MLP dec，再 sigmoid，与 dec(tp_cur) 相乘（写入新记忆）
            tp_remember = F.sigmoid(self.memory_temporal[idx](tp_cur).permute(0,2,1)) * dec(tp_cur).permute(0,2,1)
            # 原型更新为 recall + remember 的融合
            tp = tp_recall + tp_remember
            # 对原型维做平均，得到样本级时间表征 (B,256)
            tp_list.append(tp.mean(2))
        # 投影到 768 并归一化
        tp_proj_list = []
        for tp_ele in tp_list:
            tp_proj_list.append(F.normalize(self.temporal_project(tp_ele), p=2, dim=1))

        # semantic
        # ===== 与语义原型（多粒度）进行匹配 =====
        # 输入的 spatial_round / temporal_round 形状注释：list 长度为 3（粗/中/细）
        # spatial_round[k]：形状 (C, K, 768)，如 C=类数(55或5等)，K=top-k 语义块（此处注释里写 10）
        # 做 L2 归一化，便于用点积近似余弦相似度
        spatial_fg_norm = F.normalize(spatial_round[2], p=2, dim=-1)   # 55(5) 10 768
        spatial_mg_norm = F.normalize(spatial_round[1], p=2, dim=-1)   # 55(5) 10 768
        spatial_cg_norm = F.normalize(spatial_round[0], p=2, dim=-1)   # 55(5) 10 768
        spatial_sem_norm_list = [spatial_cg_norm, spatial_mg_norm, spatial_fg_norm]
        temporal_fg_norm = F.normalize(temporal_round[2], p=2, dim=-1)   # 55(5) 10 768
        temporal_mg_norm = F.normalize(temporal_round[1], p=2, dim=-1)   # 55(5) 10 768
        temporal_cg_norm = F.normalize(temporal_round[0], p=2, dim=-1)   # 55(5) 768
        temporal_sem_norm_list = [temporal_cg_norm, temporal_mg_norm, temporal_fg_norm]

        # multiply
        # ===== 空间分支：与语义原型做相似度并汇聚 =====
        logits_spatial_list = []
        # 计算相似度：'nd,ckd->nck'，样本表征 (B,768) 与语义 (C,K,768) 做点积 -> (B,C,K)
        # 对 K 维取 topk(10) 后均值，得到每类得分 (B,C)
        logits_spatial_cg = torch.einsum('nd,ckd->nck', sp_proj_list[0], spatial_cg_norm).topk(10, dim=2)[0].mean(2)  # top3
        logits_spatial_mg = torch.einsum('nd,ckd->nck', sp_proj_list[1], spatial_mg_norm).topk(10, dim=2)[0].mean(2)  # n 55 10
        logits_spatial_fg = torch.einsum('nd,ckd->nck', sp_proj_list[2], spatial_fg_norm).topk(10, dim=2)[0].mean(2)

        # 加上比例系数（0.1）作为权重，存入列表（可视为多粒度融合）
        logits_spatial_list.append(logits_spatial_cg*0.1)  # n 55  ntu60-0.1
        logits_spatial_list.append(logits_spatial_mg*0.1)  # n 55
        logits_spatial_list.append(logits_spatial_fg*0.1)  # n 55

        # ===== 时间分支：同空间分支操作 =====
        logits_temporal_list = []
        logits_temporal_cg = torch.einsum('nd,ckd->nck', tp_proj_list[0], temporal_cg_norm).topk(10, dim=2)[0].mean(2)
        logits_temporal_mg = torch.einsum('nd,ckd->nck', tp_proj_list[1], temporal_mg_norm).topk(10, dim=2)[0].mean(2) # n 55 10
        logits_temporal_fg = torch.einsum('nd,ckd->nck', tp_proj_list[2], temporal_fg_norm).topk(10, dim=2)[0].mean(2)
        logits_temporal_list.append(logits_temporal_cg*0.1)  # n 55
        logits_temporal_list.append(logits_temporal_mg*0.1)  # n 55
        logits_temporal_list.append(logits_temporal_fg*0.1)  # n 55

        # 返回三个粒度下的空间/时间 logits 列表（每个元素形状均为 (B,C)）
        return logits_spatial_list, logits_temporal_list