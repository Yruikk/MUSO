import random
import torch
from torchvision import datasets, transforms
import scipy.io
import numpy as np
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm, det, cond, matrix_rank, eig
from tqdm import *
from sklearn.metrics.pairwise import cosine_similarity


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def SGDtraining(X, y, w_init, epoch):
    wt1 = np.copy(w_init)
    err_list = []
    for id in tqdm(range(epoch * y.shape[0])):
        t = id % y.shape[0]
        gammat = 1 * pow(t + 1, -0.2)
        xt = X[t:t + 1, :]
        grad = 2 * xt.T @ (xt @ wt1 - y[t])  # + 2 * 1e-8 * wt1
        wt = wt1 - gammat * grad
        y_pred = X @ wt
        check = norm(y_pred - y)
        if check < 1e-6:
            print('break in', id)
            break
        err_list.append(check)
        wt1 = np.copy(wt)

    err_list = np.array(err_list).reshape(-1, 1)

    return wt1, err_list


def SGDtrainingStop(X, y, w_init, epoch, y_u, f_acc):
    N_u = y_u.shape[0]
    N_r = y.shape[0] - N_u
    wt1 = np.copy(w_init)
    err_list = []
    for id in tqdm(range(epoch * y.shape[0])):
        t = id % y.shape[0]
        gammat = 1 * pow(t + 1, -0.2)
        xt = X[t:t + 1, :]
        grad = 2 * xt.T @ (xt @ wt1 - y[t])  # + 2 * 1e-8 * wt1
        wt = wt1 - gammat * grad
        y_pred = X[N_r:, :] @ wt
        acc_u = np.mean(np.sign(y_pred) == y_u)

        if (np.abs(acc_u - f_acc)) < 0.03:
            print('early stop in', id)
            wt1 = np.copy(wt)
            break
        check = norm(X @ wt - y)
        if check < 1e-6:
            print('break in', id)
            wt1 = np.copy(wt)
            break
        err_list.append(check)
        wt1 = np.copy(wt)

    err_list = np.array(err_list).reshape(-1, 1)

    return wt1, err_list


def calculate_acc_retain_forg_test(X_r, X_u, X_test, w, y_r, y_u, y_test):
    y_r_pred = np.sign(X_r @ w)
    y_u_pred = np.sign(X_u @ w)
    y_test_pred = np.sign(X_test @ w)
    acc_r = np.mean(y_r_pred == y_r)
    acc_u = np.mean(y_u_pred == y_u)
    acc_test = np.mean(y_test_pred == y_test)

    return acc_r, acc_u, acc_test


# 导入MAT文件
mat_data = scipy.io.loadmat('mnist_37.mat')
data = mat_data['X_train']
raw_X_all = mat_data['X_train']  # N x d
raw_y_all = mat_data['Y_train']  # N x 1
raw_X_test = mat_data['X_test']
raw_y_test = mat_data['Y_test']

retrain_fullClass_TA, retrain_fullClass_FA = [], []
pretrain_fullClass_TA, pretrain_fullClass_FA, pretrain_fullClass_Delta = [], [], []
Am_fullClass_TA, Am_fullClass_FA, Am_fullClass_Delta = [], [], []
BT_fullClass_TA, BT_fullClass_FA, BT_fullClass_Delta = [], [], []
Ours_fullClass_TA, Ours_fullClass_FA, Ours_fullClass_Delta = [], [], []

retrain_subClass_TA, retrain_subClass_FA = [], []
pretrain_subClass_TA, pretrain_subClass_FA, pretrain_subClass_Delta = [], [], []
Am_subClass_TA, Am_subClass_FA, Am_subClass_Delta = [], [], []
BT_subClass_TA, BT_subClass_FA, BT_subClass_Delta = [], [], []
Ours_subClass_TA, Ours_subClass_FA, Ours_subClass_Delta = [], [], []

retrain_random_TA, retrain_random_FA = [], []
pretrain_random_TA, pretrain_random_FA, pretrain_random_Delta = [], [], []
Am_random_TA, Am_random_FA, Am_random_Delta = [], [], []
BT_random_TA, BT_random_FA, BT_random_Delta = [], [], []
Ours_random_TA, Ours_random_FA, Ours_random_Delta = [], [], []

D = 2500
sigma = 20
epsilon = 1e-8
epoch = 1500
for i in range(5):
    # seed_torch(i)
    # RFF.
    N_all, N_test, d = np.size(raw_X_all, 0), np.size(raw_X_test, 0), np.size(raw_X_all, 1)
    permutation_all = np.arange(N_all)
    permutation_test = np.arange(N_test)

    X_all = raw_X_all[permutation_all]
    y_all = raw_y_all[permutation_all]
    X_test = raw_X_test[permutation_test]
    y_test = raw_y_test[permutation_test]


    W = 1 / sigma * np.random.randn(d, D)
    Z_all = np.sqrt(1 / D) * np.concatenate((np.cos(X_all @ W), np.sin(X_all @ W)), axis=1)
    Z_test = np.sqrt(1 / D) * np.concatenate((np.cos(X_test @ W), np.sin(X_test @ W)), axis=1)

    N_r = 300
    N_u = N_all - N_r
    Z_r = Z_all[:N_r, :]
    y_r = y_all[:N_r, :]
    Z_u = Z_all[N_r:, :]
    y_u = y_all[N_r:, :]

    # The best solution for retain set.
    w_r_opt = Z_r.T @ inv(Z_r @ Z_r.T + epsilon * np.eye(N_r)) @ y_r
    w_init = w_r_opt + np.random.randn(2 * D, 1)
    err_r_opt = norm(Z_r @ w_r_opt - y_r) ** 2 / y_r.shape[0]
    y_pred_r_opt = np.sign(Z_test @ w_r_opt)

    w_r, r_list = SGDtraining(Z_r, y_r, w_init, epoch)
    _, wr_acc_u, wr_acc_test = calculate_acc_retain_forg_test(Z_r, Z_u, Z_test, w_r, y_r, y_u, y_test)

    w_p, p_list = SGDtraining(Z_all, y_all, w_init, epoch)
    _, wp_acc_u, wp_acc_test = calculate_acc_retain_forg_test(Z_r, Z_u, Z_test, w_p, y_r, y_u, y_test)

    hat_y_u = Z_u @ Z_r.T @ inv(Z_r @ Z_r.T + epsilon * np.eye(N_r)) @ (y_r - Z_r @ w_init) + Z_u @ w_init
    tilde_Z_all = np.concatenate((Z_r, Z_u), axis=0)
    tilde_y_all = np.concatenate((y_r, hat_y_u), axis=0)
    w_u, u_list = SGDtraining(tilde_Z_all, tilde_y_all, w_p, epoch)
    _, wu_acc_u, wu_acc_test = calculate_acc_retain_forg_test(Z_r, Z_u, Z_test, w_u, y_r, y_u, y_test)

    # random label
    random_array = np.random.randint(2, size=N_u)
    hat_y_u_rl = np.where(random_array == 0, -1, 1).reshape(-1, 1)
    tilde_y_all_rl = np.concatenate((y_r, hat_y_u_rl), axis=0)
    w_u2, u2_list = SGDtrainingStop(tilde_Z_all, tilde_y_all_rl, w_p, epoch, y_u, wr_acc_u)
    _, wu2_acc_u, wu2_acc_test = calculate_acc_retain_forg_test(Z_r, Z_u, Z_test, w_u2, y_r, y_u, y_test)

    # bad teacher
    w_bd = np.random.randn(2 * D, 1)
    hat_y_u_bd = Z_u @ w_bd
    tilde_y_all_bd = np.concatenate((y_r, hat_y_u_bd), axis=0)
    w_u3, u3_list = SGDtrainingStop(tilde_Z_all, tilde_y_all_bd, w_p, epoch, y_u, wr_acc_u)
    _, wu3_acc_u, wu3_acc_test = calculate_acc_retain_forg_test(Z_r, Z_u, Z_test, w_u3, y_r, y_u, y_test)

    d_rp = w_r - w_p
    d_ru = w_r - w_u
    d2_ru = w_r - w_u2
    d3_ru = w_r - w_u3

    retrain_fullClass_TA.append(wr_acc_test * 100)
    retrain_fullClass_FA.append(wr_acc_u * 100)
    pretrain_fullClass_TA.append(wp_acc_test * 100)
    pretrain_fullClass_FA.append(wp_acc_u * 100)
    pretrain_fullClass_Delta.append(norm(d_rp) ** 2 / (2 * D))
    Am_fullClass_TA.append(wu2_acc_test * 100)
    Am_fullClass_FA.append(wu2_acc_u * 100)
    Am_fullClass_Delta.append(norm(d2_ru) ** 2 / (2 * D))
    BT_fullClass_TA.append(wu3_acc_test * 100)
    BT_fullClass_FA.append(wu3_acc_u * 100)
    BT_fullClass_Delta.append(norm(d3_ru) ** 2 / (2 * D))
    Ours_fullClass_TA.append(wu_acc_test * 100)
    Ours_fullClass_FA.append(wu_acc_u * 100)
    Ours_fullClass_Delta.append(norm(d_ru) ** 2 / (2 * D))

print('\n')

print('fullClass TA:')
print(np.mean(retrain_fullClass_TA), '\t', np.mean(pretrain_fullClass_TA), '\t', np.mean(Am_fullClass_TA), '\t', np.mean(BT_fullClass_TA), '\t', np.mean(Ours_fullClass_TA))
print(np.std(retrain_fullClass_TA), '\t', np.std(pretrain_fullClass_TA), '\t', np.std(Am_fullClass_TA), '\t', np.std(BT_fullClass_TA), '\t', np.std(Ours_fullClass_TA))
print('fullClass FA:')
print(np.mean(retrain_fullClass_FA), '\t', np.mean(pretrain_fullClass_FA), '\t', np.mean(Am_fullClass_FA), '\t', np.mean(BT_fullClass_FA), '\t', np.mean(Ours_fullClass_FA))
print(np.std(retrain_fullClass_FA), '\t', np.std(pretrain_fullClass_FA), '\t', np.std(Am_fullClass_FA), '\t', np.std(BT_fullClass_FA), '\t', np.std(Ours_fullClass_FA))
print('fullClass Delta:')
print('\t', np.mean(pretrain_fullClass_Delta), '\t', np.mean(Am_fullClass_Delta), '\t', np.mean(BT_fullClass_Delta), '\t', np.mean(Ours_fullClass_Delta))
print('\t', np.std(pretrain_fullClass_Delta), '\t', np.std(Am_fullClass_Delta), '\t', np.std(BT_fullClass_Delta), '\t', np.std(Ours_fullClass_Delta))

pass