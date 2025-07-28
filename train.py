import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import argparse
from networks.GMVAE_OG import GMVAE
import dataloader as dl
import numpy as np
from sklearn.metrics import f1_score



parser = argparse.ArgumentParser(description='Gaussian Mixture VAE')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--K', type=int, metavar='N',
                    help='number of clusters')
parser.add_argument('--x-size', type=int, default=200, metavar='N',
                    help='dimension of x')
parser.add_argument('--hidden-size', type=int, default=512, metavar='N',
                    help='dimension of hidden layer')
parser.add_argument('--w-size', type=int, default=128, metavar='N',
                    help='dimension of w')
parser.add_argument('--dataset', help='dataset to use')
parser.add_argument('--feature-type', default='c3d', help='dataset to use')
parser.add_argument('--learning-rate', type=float, default=1e-4,
                    help='learning rate for optimizer')
parser.add_argument('--continuous', help='data is continuous',
					action='store_true')

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
args.dataset = 'Adult'
args.K = 10

if args.dataset == 'Adult':
     train_loader , test_loader  = dl.Adultloader(args.batch_size)

if args.dataset == 'Adult':
    gmvae = GMVAE(args).to(device)


optimizer = optim.Adam(gmvae.parameters(), lr=args.learning_rate)
lam = torch.Tensor([100.0]).to(device)

def loss_function(output,train_sample,mask1s, mu_w, logvar_w,qz,
                  mu_x, logvar_x, mu_px, logvar_px, likelihood):

    output = output.float()
    images = train_sample.float()
    mask1s=mask1s[:,:6]
    numeric_output = output[:, :6]
    categorical_output = output[:, 6:]
    numeric_images = images[:, :6]
    categorical_images = images[:, 6:]

    numeric_loss = F.binary_cross_entropy(numeric_output, numeric_images, reduction='sum')

    categories_count = [7, 16, 7, 14, 6, 5, 2, 41]
    loss_fn = nn.CrossEntropyLoss()

    start_idx = 0
    total_loss = 0
    for feature_idx, num_classes in enumerate(categories_count):
        end_idx = start_idx + num_classes
        logits = categorical_output[:, start_idx:end_idx]
        target = categorical_images[:, feature_idx]
        target = target.long()
        target = target - 1
        loss = loss_fn(logits, target)
        total_loss += loss
        start_idx = end_idx
    recon_loss=total_loss+numeric_loss

    # 2. KL( q(w) || p(w) )
    KLD_W = -0.5 * torch.sum(1 + logvar_w - mu_w.pow(2) - logvar_w.exp())# 散度（KLD）
    # 3. KL( q(z) || p(z) )
    KLD_Z = torch.sum(qz * torch.log(args.K * qz + 1e-10))
    if args.dataset == 'spiral':
        KLD_Z = max(lam, KLD_Z)

    # 4. E_z_w[KL(q(x)|| p(x|z,w))]
    mu_x = mu_x.unsqueeze(-1)

    logvar_x = logvar_x.unsqueeze(-1)
    logvar_x = logvar_x.expand(-1, args.x_size, args.K)
    KLD_QX_PX = 0.5 * (((logvar_px - logvar_x) + \
                        ((logvar_x.exp() + (mu_x - mu_px).pow(2)) / logvar_px.exp())) \
                       - 1)

    qz = qz.unsqueeze(-1)
    qz = qz.expand(-1, args.K, 1)

    E_KLD_QX_PX = torch.sum(torch.bmm(KLD_QX_PX, qz))


    loss = recon_loss+KLD_W+ E_KLD_QX_PX+KLD_Z+(-likelihood)
    return loss, recon_loss, total_loss, numeric_loss, KLD_W, E_KLD_QX_PX,KLD_Z



def train(epoch):
    gmvae.train()

    for batch_idx, (train_sample, train_mask_or, final_train,final_train_or) in enumerate(train_loader):
        final_train = final_train.float()  # 转换数据类型
        final_train = final_train.to(device)
        final_train_or = final_train_or.to(device)
        train_sample=train_sample.to(device)
        train_mask_or=train_mask_or.to(device)

        optimizer.zero_grad()

        mu_x, logvar_x, mu_px, logvar_px,qz, output, mu_w, logvar_w, \
          likelihood = gmvae(final_train,final_train_or)
        loss, recon_loss,  total_loss, numeric_loss, KLD_W, E_KLD_QX_PX, KLD_Z\
            = loss_function(output,train_sample,train_mask_or, mu_w, logvar_w,qz,
                            mu_x, logvar_x, mu_px, logvar_px, likelihood)

        loss.backward()
        optimizer.step()

    mask1s = train_mask_or[:, :6]
    numeric_output = output[:, :6]
    categorical_output = output[:, 6:]
    numeric_images = train_sample[:, :6]
    categorical_images = train_sample[:, 6:]
    categories_count = [7, 16, 7, 14, 6, 5, 2, 41]
    # # 逐列计算 F1 分数
    categorical_images = categorical_images.detach().cpu().numpy()
    categorical_output = categorical_output.detach().cpu().numpy()
    all_categorical_true = []
    all_categorical_pred = []

    # 遍历每个特征计算 F1 分数
    for i in range(len(categories_count)):
        start = sum(categories_count[:i])
        end = sum(categories_count[:i + 1])

        categorical_pred = np.argmax(categorical_output[:, start:end], axis=-1)
        categorical_true = categorical_images[:, i] - 1

        # 将每个特征的预测和真实标签拼接
        all_categorical_true.append(categorical_true)
        all_categorical_pred.append(categorical_pred)

    all_categorical_true = np.concatenate(all_categorical_true, axis=0)
    all_categorical_pred = np.concatenate(all_categorical_pred, axis=0)

    # 计算 micro F1 分数
    global_f1 = f1_score(all_categorical_true, all_categorical_pred, average='micro')


    # RMSE
    diff = numeric_output - numeric_images
    squared_diff = diff.pow(2)
    mse = torch.mean(squared_diff)
    rmse = torch.sqrt(mse)
    # observed RMSE
    diff_obs = (numeric_output - numeric_images) * mask1s
    squared_diff_obs = diff_obs.pow(2)
    squared_diff_obs = torch.sum(squared_diff_obs) / torch.sum(mask1s)
    rmse_obs = torch.sqrt(squared_diff_obs)

    # miss RMSE
    diff_mask_miss = (numeric_output - numeric_images) * (1 - mask1s)
    squared_diff_miss = diff_mask_miss.pow(2)
    squared_diff_miss = torch.sum(squared_diff_miss) / torch.sum((1 - mask1s))
    rmse_miss = torch.sqrt(squared_diff_miss)


    print("epoch: " + str(epoch) + " loss: " + str(loss.item())  + "recon_loss: " + str(recon_loss.item()) + "total_loss: " + str(total_loss.item()) + "numeric_loss: " + str(numeric_loss.item()) + "KLD_W: " + str(KLD_W.item()) +" E_KLD_QX_PX: " + str(E_KLD_QX_PX.item())+"KLD_Z: " + str(KLD_Z.item()) + " rmse: " + str(rmse.item())+ " rmse_obs: " + str(
            rmse_obs.item())+  " rmse_miss: " + str(rmse_miss.item())+ " likelihood: " + str(likelihood.item())+ " f1: " + str(global_f1))


model_file = './model/' + args.dataset + '_' + str(args.K) + 'KACGI.pth'



for epoch in range(1, args.epochs + 1):
    train(epoch)

    if epoch  == 200:
        torch.save(gmvae.state_dict(), model_file)

def test_mean(num_obs_mean,num_miss_mean,global_F):

    model_file_mean = './model/' + args.dataset + '_' + str(args.K) + 'KACGI.pth'
    gmvae.load_state_dict(torch.load(model_file_mean))

    gmvae.eval()

    with torch.no_grad():
        total_squared_diff_observed_mean = 0
        total_squared_diff_missing_mean = 0
        total_pixels_observed_mean = 0
        total_pixels_missing_mean = 0

        all_categorical_true_missing = []
        all_categorical_pred_missing = []

        for batch_idx, (test_sample, test_mask_or, final_test,final_test_or) in enumerate(test_loader):
            final_test = final_test.float()
            final_test = final_test.to(device)
            final_test_or = final_test_or.to(device)
            test_sample = test_sample.to(device)
            mask1s = test_mask_or[:, :6].to(device)
            mask = test_mask_or[:, 6:].to(device)
            mu_x, logvar_x, mu_px, logvar_px, qz, output, mu_w, \
            logvar_w, log_likelihood = gmvae(final_test,final_test_or)
            numeric_output = output[:, :6]
            categorical_output = output[:, 6:]
            numeric_images = test_sample[:, :6]
            categorical_images = test_sample[:, 6:]

            # 观察值
            diff_observed = (numeric_images - numeric_output) * mask1s
            squared_diff_observed = diff_observed.pow(2)
            total_squared_diff_observed_mean += torch.sum(squared_diff_observed)
            total_pixels_observed_mean += torch.sum(mask1s).item()

            # 缺失值
            diff_missing = (numeric_images - numeric_output) * (1 - mask1s)
            squared_diff_missing = diff_missing.pow(2)
            total_squared_diff_missing_mean += torch.sum(squared_diff_missing)
            total_pixels_missing_mean += torch.sum(1 - mask1s).item()


            categories_count = [7, 16, 7, 14, 6, 5, 2, 41]  # 每个特征的类别数
            # # 逐列计算 F1 分数
            categorical_images = categorical_images.detach().cpu().numpy()
            categorical_output = categorical_output.detach().cpu().numpy()
            # 遍历每个特征计算 F1 分数
            for i in range(len(categories_count)):
                start = sum(categories_count[:i])
                end = sum(categories_count[:i + 1])

                categorical_pred = np.argmax(categorical_output[:, start:end], axis=-1)
                categorical_true = categorical_images[:, i] - 1
                categorical_mask = mask[:, i]
                categorical_mask = categorical_mask.cpu().numpy()

                categorical_true_missing = categorical_true * (1 - categorical_mask)  # 真实标签与 mask 相乘，缺失部分为 0
                categorical_pred_missing = categorical_pred * (1 - categorical_mask)  # 预测标签与 mask 相乘，缺失部分为 0


                all_categorical_true_missing.append(categorical_true_missing)
                all_categorical_pred_missing.append(categorical_pred_missing)

        mse_observed = total_squared_diff_observed_mean / total_pixels_observed_mean
        rmse_observed = torch.sqrt(mse_observed)
        num_obs_mean.append(rmse_observed)
        print(f"第 {test_idx + 1} 次测试观察mean RMSE:", rmse_observed.item())

        # 计算缺失值部分的整体 RMSE
        mse_missing = total_squared_diff_missing_mean / total_pixels_missing_mean
        rmse_missing = torch.sqrt(mse_missing)
        num_miss_mean.append(rmse_missing)
        print(f"第 {test_idx + 1} 次测试缺失mean RMSE:", rmse_missing.item())


        # 将所有批次的缺失部分真实标签和预测标签连接成一个大数组
        all_categorical_true_missing = np.concatenate(all_categorical_true_missing, axis=0)
        all_categorical_pred_missing = np.concatenate(all_categorical_pred_missing, axis=0)
        f1 = f1_score(all_categorical_true_missing, all_categorical_pred_missing, average='micro')
        global_F.append(f1)
        print(f"第 {test_idx + 1} 次测试缺失:", f1.item())

def mean_and_variance(rmse_list):
    mean_rmse = sum(rmse_list) / len(rmse_list)
    variance_rmse = sum((x - mean_rmse) ** 2 for x in rmse_list) / len(rmse_list)
    return mean_rmse, variance_rmse

num_obs_mean=[]
num_miss_mean=[]
num_obs_T = []
num_miss_T = []
global_F = []

for test_idx in range(20):

    test_mean(num_obs_mean, num_miss_mean,global_F)


mean_rmse_obs_mean, variance_rmse_obs_mean = mean_and_variance(num_obs_mean)
mean_rmse_miss_mean, variance_rmse_miss_mean = mean_and_variance(num_miss_mean)
mean_miss_F, variance_miss_F = mean_and_variance(global_F)
# 输出结果

print(f"Adult测试观察值 RMSE 的平均值: {mean_rmse_obs_mean}, 方差: {variance_rmse_obs_mean}")
print(f"Adult测试缺失值 RMSE 的平均值: {mean_rmse_miss_mean}, 方差: {variance_rmse_miss_mean}")
print(f"Adult测试缺失值 F 的平均值: {mean_miss_F}, 方差: {variance_miss_F}")



