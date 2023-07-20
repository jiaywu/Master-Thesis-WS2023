import numpy as np
import torch
import torch.nn as nn

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # pi


def array_response(x, N):
    a = torch.empty(N, dtype=torch.complex64)
    for i in range(N):
        a[i] = torch.cos(torch.pi * torch.sin(x) * i) + 1j * torch.sin(torch.pi * torch.sin(x) * i)

    return (1 / N ** 0.5) * a


Nr = 4
Nt = 4
SNR = torch.tensor([-10, 0, 10, 20])
n_PL = 2
sigma_alpha = (Nr * Nt) ** 0.5
sigma_obs_n = sigma_alpha * (10 ** (SNR / -20))

n_it = 5
ind_length = 100 + 1

[x_true, sensor_target, training_target, training_input, training_input_no_off, prior_lowdB, prior_middB, prior_highdB, miu_k, A_k, Q_k, R_k, R_k_no_off, r_x, r_y, r_omega] = torch.load('test_data.pt')


all_MSE_gs_low = torch.tensor([0.0, 0.0, 0.0, 0.0])
pos_low = torch.zeros(size=[n_it, 4, 3, ind_length])

all_RMSE_gs_low = torch.tensor([0.0, 0.0, 0.0, 0.0])

all_loss_gs_low = torch.tensor([0.0, 0.0, 0.0, 0.0])

MSE_gs_low = torch.zeros(3, 4)

RMSE_gs_low = torch.zeros(3, 4)

iter = 10
# ind = torch.randint(0, 1000, (1,))
ind = 1
print("i: ", ind)
num = 0

R = torch.zeros(4, 3, 3) # R matrxi for LKF with grid search MSE
R[0, ::] = torch.tensor([[0.8881 ** 2, 0.0, 0.0], [0.0, 0.9273 ** 2, 0.0], [0.0, 0.0, 0.6689 ** 2]])
R[1, ::] = torch.tensor([[0.7035 ** 2, 0.0, 0.0], [0.0, 0.8365 ** 2, 0.0], [0.0, 0.0, 0.6542 ** 2]])
R[2, ::] = torch.tensor([[0.6058 ** 2, 0.0, 0.0], [0.0, 0.8198 ** 2, 0.0], [0.0, 0.0, 0.4463 ** 2]])
R[3, ::] = torch.tensor([[0.2066 ** 2, 0.0, 0.0], [0.0, 1.0620 ** 2, 0.0], [0.0, 0.0, 0.7906 ** 2]])


for j in range(4):
    pos_low[num, j, :, 0] = x_true[:, 0]
    Sigma_x = Q_k[num, :, :, 1].squeeze()

    m1x_posterior = x_true[:, 0]
    m2x_posterior = torch.zeros(3, 3)
    # pos_low[num, j, :, 1] = torch.matmul(A_k[num, :, :, 1], training_target[num, :, 0])
    for i in range(1, ind_length):
        # target = torch.tensor([training_target[num, 0, i], training_target[num, 1, i], training_target[num, 2, i]])
        target = training_target[num, :, i]
        posterior = (torch.matmul(A_k[num, :, :, i], pos_low[num, j, :, 0]) + miu_k[num, :, i]).squeeze()

        A = A_k[num, :, :, i].squeeze()
        miu = miu_k[num, :, i].squeeze()
        Sigma_x = Q_k[num, :, :, i].squeeze()

        h_est = ((posterior[0]) ** 2) * (
                torch.abs(torch.dot(torch.conj(array_response((pos_low[num, j, 2, i-1] / 180 * torch.pi), Nr)),
                                    array_response((posterior[2] / 180 * torch.pi), Nr))) ** 2) \
                * (torch.abs(torch.dot(torch.conj(array_response((pos_low[num, j, 1, i] / 180 * torch.pi), Nt)),
                                       array_response((posterior[1] / 180 * torch.pi), Nt))) ** 2)

        tres = torch.abs(training_input_no_off[num, j, i] - h_est) ** 2
        # tres = 10000

        center = posterior
        ar = torch.linspace(center[0].item() - Sigma_x[0, 0] * 1.5, center[0].item() + Sigma_x[0, 0] * 1.5, steps=31)
        aodr = torch.linspace(center[1].item() - Sigma_x[1, 1] * 1.5, center[1].item() + Sigma_x[1, 1] * 1.5, steps=31)
        aoar = torch.linspace(center[2].item() - Sigma_x[2, 2] * 1.5, center[2].item() + Sigma_x[2, 2] * 1.5, steps=31)


        p_ar = 0
        p_ad = 0
        p_aa = 0
        for i_ar in range(len(ar)):
            if ar[i_ar] < 0:
                ar[i_ar] = 0

            for i_ad in range(len(aodr)):
                # if aodr[i_ad] < -torch.pi:
                #    aodr[i_ad] = -torch.pi
                # if aodr[i_ad] > torch.pi:
                #    aodr[i_ad] = torch.pi

                for i_aa in range(len(aoar)):
                    # if aoar[i_aa] < -torch.pi:
                    #    aoar[i_aa] = -torch.pi
                    # if aoar[i_aa] > torch.pi:
                    #    aoar[i_aa] = torch.pi

                    h_test = ((ar[i_ar]) ** 2) * (
                            torch.abs(torch.dot(torch.conj(array_response((pos_low[num, j, 2, i-1] / 180 * torch.pi), Nr)),
                                                array_response((aoar[i_aa] / 180 * torch.pi), Nr))) ** 2) \
                             * (torch.abs(torch.dot(torch.conj(array_response((aodr[i_ad] / 180 * torch.pi), Nt)),
                                                    array_response(
                                                        ((pos_low[num, j, 1, i-1] / 180 * torch.pi)),
                                                        Nt))) ** 2)
                    loss = torch.abs(training_input_no_off[num, j, i] - h_test) ** 2
                    if loss < tres:
                        posterior = torch.tensor([ar[i_ar], aodr[i_ad], aoar[i_aa]])
                        tres = loss
                        p_ar = i_ar
                        p_ad = i_ad
                        p_aa = i_aa
                    # print(j, i, i_ar, i_ad, i_aa)

        # pos_low[num, j, :, i] = posterior

        # m1x_prior = posterior
        m1x_prior = torch.matmul(A, m1x_posterior) + miu
        # m1x_prior = m1x_prior + miu
        m2x_prior = torch.matmul(A, m2x_posterior)
        m2x_prior = torch.matmul(m2x_prior, torch.transpose(A, 0, 1)) + Sigma_x

        # R = torch.tensor([[r1, 0.0, 0.0], [0.0, r2, 0.0], [0.0, 0.0, r3]])

        KG_inter = torch.matmul(m2x_prior, torch.eye(3))
        KG_inter = torch.matmul(torch.eye(3), KG_inter) + R[j, ::].squeeze()
        KG_inter = torch.linalg.inv(KG_inter)
        KG = torch.matmul(m2x_prior, torch.eye(3))
        KG = torch.matmul(KG, KG_inter)

        m1x_posterior = posterior + torch.matmul(KG, (posterior - m1x_prior))  # .to(torch.float)
        m2x_posterior = torch.matmul((torch.eye(3) - KG), m2x_prior)  # .float()

        pos_low[num, j, :, i] = m1x_posterior

        print("i: ", i, "ground truth low2: ", target)
        print("i: ", i, "GS estimated: ", posterior)
        print("i: ", i, "estimated: ", m1x_posterior)
        print("i: ", i, "loss: ", tres)

        all_MSE_gs_low[j] += torch.norm(target - m1x_posterior) ** 2
        all_loss_gs_low[j] += tres
        MSE_gs_low[0, j] += torch.abs(target[0] - m1x_posterior[0]) ** 2
        MSE_gs_low[1, j] += torch.abs(target[1] - m1x_posterior[1]) ** 2
        MSE_gs_low[2, j] += torch.abs(target[2] - m1x_posterior[2]) ** 2

    MSE_gs_low[:, j] = MSE_gs_low[:, j] / iter
    all_MSE_gs_low[j] = all_MSE_gs_low[j] / iter
    all_loss_gs_low[j] = all_loss_gs_low[j] / iter


RMSE_gs_low = MSE_gs_low ** 0.5

all_RMSE_gs_low = all_MSE_gs_low ** 0.5

MSE_gs_low = 10 * torch.log10(MSE_gs_low)

all_MSE_gs_low = 10 * torch.log10(all_MSE_gs_low)

print("grid search, 1/r^2 = -10dB, MSE:", MSE_gs_low)

print(all_MSE_gs_low)

print("grid search, 1/r^2 = -10dB, RMSE:", RMSE_gs_low)

print(all_RMSE_gs_low)

print(all_loss_gs_low)


