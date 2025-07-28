import torch
from torch import nn
from torch.nn import functional as F
import math
import torchvision.transforms as tt
from torchvision.utils import save_image
import os
import numpy as np






class GMVAE(nn.Module):
    def __init__(self, args,num_capsules=24, template_size=11, num_templates=24,num_feature_maps=24):
        super(GMVAE, self).__init__()
        self.device = args.device
        #胶囊网络
        self.num_capsules = num_capsules
        self.num_feature_maps = num_feature_maps
        # 顺序组合多个层或操作Sequential
        self.capsules = nn.Sequential(nn.Conv2d(1, 128, (3,5), stride=(1,2)),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 128, (3,5), stride=(1,2)),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 128, (3,5), stride=(1,2)),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 128, (3,5), stride=(1,2)),
                                      nn.ReLU(),
                                      nn.Conv2d(128, num_capsules * num_feature_maps, (2,3), stride=1))


        self.h1 = nn.Linear(2304, 500, bias=False)
        self.b4 = nn.BatchNorm1d(500)
        self.mu_x = nn.Linear(500, 200)
        self.logvar_x = nn.Linear(500, 200)
        self.mu_w = nn.Linear(500, 150)
        self.logvar_w = nn.Linear(500, 150)
        self.qz = nn.Linear(500, 10)

        # prior generator
        self.h2 = nn.Linear(150, 500)
        self.mu_px = nn.ModuleList(
            [nn.Linear(500, 200) for i in range(10)])
        self.logvar_px = nn.ModuleList(
            [nn.Linear(500, 200) for i in range(10)])




        self.h3 = nn.Linear(29768, 500, bias=False)
        self.b5 = nn.BatchNorm1d(500)
        self.h4 = nn.Linear(500, 31680, bias=False)
        self.b6 = nn.BatchNorm1d(31680)
        self.d3 = nn.ConvTranspose2d(576, 32, (3,5), (1,2), 0, bias=False)# 5,19
        self.b7 = nn.BatchNorm2d(32)
        self.d2 = nn.ConvTranspose2d(32, 16, (3,7), (1,2), 0, bias=False)
        self.b8 = nn.BatchNorm2d(16)
        self.d1 = nn.ConvTranspose2d(16, 1, (3,4), (1,2), 0)

        self.templates = nn.ParameterList([nn.Parameter(torch.randn(1, template_size, template_size))
                                           for _ in range(num_templates)])
        self.soft_max = nn.Softmax(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.to_pil = tt.ToPILImage()
        self.to_tensor = tt.ToTensor()
        self.epsilon = torch.tensor(1e-6)

    def encode(self, X,final_train_or,mode = 'train'):
        outputs = self.capsules(X)
        x = outputs.view(-1, 2304)
        x = self.h1(x)
        x = F.relu(self.b4(x))
        qz = F.softmax(self.qz(x), dim=1)
        mu_x = self.mu_x(x)
        logvar_x = self.logvar_x(x)
        mu_w = self.mu_w(x)
        logvar_w = self.logvar_w(x)

        outputs = outputs.view(-1, self.num_capsules, self.num_feature_maps, *outputs.size()[2:])  # (B,M,24,2,2)
        attention = outputs[:, :, -1, :, :].unsqueeze(2)

        attention_soft = self.soft_max(attention.view(*attention.size()[:3], -1)).view_as(attention)

        feature_maps = outputs[:, :, :-1, :, :]

        part_capsule_param = torch.sum(torch.sum(feature_maps * attention_soft, dim=-1),
                                       dim=-1)  #

        if mode == 'train':
            noise_1 = torch.FloatTensor(*part_capsule_param.size()[:2]).uniform_(-2, 2).to(self.device)

        else:
            noise_1 = torch.zeros(*part_capsule_param.size()[:2]).to(self.device)  # 全0张量
        x_m, d_m, c_z = self.relu(part_capsule_param[:, :, :6]), self.sigmoid(
            part_capsule_param[:, :, 6] + noise_1).view(*part_capsule_param.size()[:2], 1), self.relu(
            part_capsule_param[:, :, 7:])

        B, _, k, target_size = X.size()

        transformed_templates = [
            F.grid_sample(
                self.templates[i].repeat(B, 1, 1, 1).to(self.device),
                F.affine_grid(
                    self.geometric_transform(x_m[:, i, :]),
                    torch.Size((B, 1, k, target_size)),
                    align_corners=True
                ),
                align_corners=True
            )
            for i in range(self.num_capsules)
        ]
        transformed_templates = torch.cat(transformed_templates,1)

        mix_prob = self.soft_max(d_m * transformed_templates.view(*transformed_templates.size()[:2], -1)).view_as(
            transformed_templates)
        detach_x = final_train_or.data

        std = detach_x.view(*X.size()[:2], -1).std(-1).unsqueeze(1)
        std = std * 1 + self.epsilon

        multiplier = (std * math.sqrt(math.pi * 2)).reciprocal().unsqueeze(-1)
        power_multiply = (-(2 * (std ** 2))).reciprocal().unsqueeze(-1)
        gaussians = multiplier * (
            (((detach_x - transformed_templates) ** 2) * power_multiply).exp())


        pre_ll = (gaussians * mix_prob * 1.0) + self.epsilon
        log_likelihood = torch.sum(pre_ll, dim=1).log().sum(-1).sum(-1).mean()

        return qz, mu_x, logvar_x, mu_w, logvar_w, transformed_templates,log_likelihood

    @staticmethod
    def geometric_transform(pose_tensor, similarity=False, nonlinear=True):  # 相似性变换（旋转、缩放和平移),非线性变换


        scale_x, scale_y, theta, shear, trans_x, trans_y = torch.split(pose_tensor, 1, -1)

        if nonlinear:
            scale_x, scale_y = torch.sigmoid(scale_x) + 1e-2, torch.sigmoid(scale_y) + 1e-2
            trans_x, trans_y, shear = torch.tanh(trans_x * 5.), torch.tanh(trans_y * 5.), torch.tanh(
                shear * 5.)
            theta = theta * 2. * math.pi
        else:
            scale_x, scale_y = (abs(i) + 1e-2 for i in (scale_x, scale_y))

        c, s = torch.cos(theta), torch.sin(theta)

        if similarity:
            scale = scale_x
            pose = [scale * c, -scale * s, trans_x, scale * s, scale * c, trans_y]

        else:
            pose = [
                scale_x * c + shear * scale_y * s, -scale_x * s + shear * scale_y * c,
                trans_x, scale_y * s, scale_y * c, trans_y
            ]

        pose = torch.cat(pose, -1)

        # convert to a matrix
        shape = list(pose.shape[:-1])
        shape += [2, 3]
        pose = torch.reshape(pose, shape)

        return pose

    def priorGenerator(self, w_sample):
        batchSize = w_sample.size(0)
        h = torch.tanh(self.h2(w_sample))
        mu_px = torch.empty(batchSize, 200, 10,
                            device=self.device, requires_grad=False)
        logvar_px = torch.empty(batchSize, 200, 10,
                                device=self.device, requires_grad=False)

        for i in range(10):
            mu_px[:, :, i] = self.mu_px[i](h)
            logvar_px[:, :, i] = self.logvar_px[i](h)

        return mu_px, logvar_px

    def decoder(self, combined):
        h = self.h3(combined)
        h = F.relu(self.b5(h))
        h = self.h4(h)
        h = F.relu(self.b6(h))
        h = h.view(-1, 576, 5, 11)
        h = self.d3(h)
        h = F.relu(self.b7(h))
        h = self.d2(h)
        h = F.relu(self.b8(h))
        h = self.d1(h)
        Y = torch.sigmoid(h)

        return Y


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, X,final_train_or):
        X=X.view(-1,1,11,112)
        final_train_or=final_train_or.view(-1,1,11,112)
        qz, mu_x, logvar_x, mu_w, logvar_w, transformed_templates, log_likelihood = self.encode(X,final_train_or)
        w_sample = self.reparameterize(mu_w, logvar_w)
        mu_px, logvar_px = self.priorGenerator(w_sample)

        Y_list = []
        numerator_list = []
        denominator = 0
        input_dim = 112
        V = nn.Parameter(torch.randn(input_dim))
        V = V.to(self.device)


        transformed_templates = transformed_templates.view(32, -1)

        for i in range(10):
            x_sample = self.reparameterize(mu_x, logvar_x)# [32,200]

            combined = torch.cat((transformed_templates, x_sample), axis=1)

            Y = self.decoder(combined)
            Y_list.append(Y)

            numerator = torch.exp(V * torch.tanh(Y))
            numerator_list.append(numerator)
            denominator += numerator


        output = 0

        for Y, numerator in zip(Y_list, numerator_list):
            w = numerator / denominator
            output += w * Y
        output = torch.clamp(output, min=0.0, max=1.0 - 1e-8)

        output = output[:, :, 0]

        output=output.view(32,-1)


        return mu_x, logvar_x, mu_px, logvar_px, qz, output, mu_w, \
            logvar_w,log_likelihood


