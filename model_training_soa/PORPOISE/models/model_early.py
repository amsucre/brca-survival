import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from os.path import join
from collections import OrderedDict

class BilinearFusion(nn.Module):
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU())
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU())
        #init_max_weights(self)

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out

def SNN_Block(dim1, dim2, dropout=0.25):
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))


def MLP_Block(dim1, dim2, dropout=0.25):
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False))


"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes (experimental usage for multiclass MIL)
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes (experimental usage for multiclass MIL)
"""
class Attn_Net_Gated(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x



def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


### MMF (in the PORPOISE Paper)
class EarlyMMF(nn.Module):
    def __init__(self, 
        omic_input_dim,
        path_input_dim=1024,
        fusion='bilinear',
        dropout=0.25,
        n_classes=4,
        scale_dim1=8,
        scale_dim2=8,
        gate_path=1,
        gate_omic=1,
        skip=True,
        dropinput=0.10,
        use_mlp=False,
        size_arg="small",
        ):
        super(EarlyMMF, self).__init__()
        self.fusion = fusion
        self.size_dict_path = {"small": [path_input_dim, 512, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}
        self.n_classes = n_classes
        self.dropinput = dropinput

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout = dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        self.omic_projector = nn.Linear(omic_input_dim, path_input_dim)  # Match patch dim

        # ### Constructing Genomic SNN
        # if self.fusion is not None:
        #     if use_mlp:
        #         Block = MLP_Block
        #     else:
        #         Block = SNN_Block
        #
        #     hidden = self.size_dict_omic['small']
        #     fc_omic = [Block(dim1=omic_input_dim, dim2=hidden[0])]
        #     for i, _ in enumerate(hidden[1:]):
        #         fc_omic.append(Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        #     self.fc_omic = nn.Sequential(*fc_omic)

        self.classifier_mm = nn.Linear(size[2], n_classes)

    def forward(self, **kwargs):
        # Step 1: Get patch features
        x_path = kwargs['x_path']  # shape: [num_patches, path_input_dim]

        # Step 2: Project omics to patch feature dim
        x_omic = kwargs['x_omic']  # shape: [batch, omic_dim]
        # x_omic_patch = self.omic_projector(x_omic)  # shape: [batch, path_input_dim]
        patch_size = 1024
        omic_chunks = torch.split(x_omic, patch_size, dim=1)

        if omic_chunks[-1].shape[1] < patch_size:
            pad_size = patch_size - omic_chunks[-1].shape[1]
            last_chunk = F.pad(omic_chunks[-1], (0, pad_size), value=0)
            omic_chunks = omic_chunks[:-1] + (last_chunk,)

        x_omic_patch = torch.stack(omic_chunks, dim=0)

        # Optional: you might want to replicate x_omic_patch to match #patches or insert just once
        x_omic_patch = x_omic_patch.squeeze()  # shape: [B, 1, path_input_dim]
        if self.training:
            if self.dropinput:
                n_patches = x_path.size(0)
                keep_ratio = 1 - self.dropinput
                n_keep = max(1, int(n_patches * keep_ratio))

                # Randomly select indices to keep
                perm = torch.randperm(n_patches, device=x_path.device)
                keep_idx = perm[:n_keep]

                # Subset the input
                x_path = x_path[keep_idx]


        # Step 3: Merge omics with image patches
        x_path = torch.cat([x_omic_patch, x_path], dim=0)  # [B, num_patches+1, path_input_dim]

        # Step 4: Run attention
        A, h_path = self.attention_net(x_path)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path)

        # Step 5: Classifier
        h_mm = self.classifier_mm(h_path)

        return h_mm
