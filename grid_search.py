import numpy as np
import torch
import torch.nn as nn
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # pi


def array_response(x, N):
    a = torch.empty(N, dtype=torch.complex64)
    for i in range(N):
        a[i] = torch.cos(torch.pi * torch.sin(x) * i) + 1j * torch.sin(torch.pi * torch.sin(x) * i)

    return (1 / N ** 0.5) * a


num = 1000
Nr = 4
Nt = 4
SNR = torch.tensor([-10, 0, 10, 20])
n_PL = 2
sigma_alpha = (Nr * Nt) ** 0.5
sigma_obs_n = sigma_alpha * (10 ** (SNR / -20))
prior_q_p = torch.tensor([2, 1, 0.5])  # ([5, 3, 1])
prior_q_a = torch.tensor([2, 1, 0.5])  # ([10, 5, 1])

n_it = 1
ind_length = 50 + 1

[x_true, sensor_target, training_target, training_input, training_input_no_off, prior_lowdB, prior_middB, prior_highdB, miu_k, A_k, Q_k, R_k, R_k_no_off, r_x, r_y, r_omega] = torch.load('test_data.pt')


all_MSE_gs_low = torch.tensor([0.0, 0.0, 0.0, 0.0])
all_MSE_gs_mid = torch.tensor([0.0, 0.0, 0.0, 0.0])
all_MSE_gs_high = torch.tensor([0.0, 0.0, 0.0, 0.0])
pos_low = torch.zeros(size=[n_it, 4, 3, ind_length])

all_RMSE_gs_low = torch.tensor([0.0, 0.0, 0.0, 0.0])
all_RMSE_gs_mid = torch.tensor([0.0, 0.0, 0.0, 0.0])
all_RMSE_gs_high = torch.tensor([0.0, 0.0, 0.0, 0.0])
pos_mid = torch.zeros(size=[n_it, 4, 3, ind_length])

all_loss_gs_low = torch.tensor([0.0, 0.0, 0.0, 0.0])
all_loss_gs_mid = torch.tensor([0.0, 0.0, 0.0, 0.0])
all_loss_gs_high = torch.tensor([0.0, 0.0, 0.0, 0.0])
pos_high = torch.zeros(size=[n_it, 4, 3, ind_length])

MSE_gs_low = torch.zeros(3, 4)
MSE_gs_mid = torch.zeros(3, 4)
MSE_gs_high = torch.zeros(3, 4)

RMSE_gs_low = torch.zeros(3, 4)
RMSE_gs_mid = torch.zeros(3, 4)
RMSE_gs_high = torch.zeros(3, 4)

iter = 10
# ind = torch.randint(0, 1000, (1,))
ind = 1
print("i: ", ind)
num = 0


# grid search low snr prior
for j in range(4):
    pos_low[num, j, :, 0] = x_true[:, 0]
    for i in range(1, ind_length):
        target = torch.tensor([training_target[num, 0, i], training_target[num, 1, i], training_target[num, 2, i]])


        h_est = ((prior_lowdB[num, 0, i]) ** 2) * (
                torch.abs(torch.dot(torch.conj(array_response((x_true[2, i-1] / 180 * torch.pi), Nr)),
                                    array_response((prior_lowdB[num, 2, i] / 180 * torch.pi), Nr))) ** 2) \
                * (torch.abs(torch.dot(torch.conj(array_response((prior_lowdB[num, 1, i] / 180 * torch.pi), Nt)),
                                       array_response((x_true[1, i-1] / 180 * torch.pi), Nt))) ** 2)
        
       
        tres = torch.abs(training_input_no_off[num, j, i] - h_est) ** 2
        #tres = 10000
        ar = torch.linspace(prior_lowdB[num, 0, i].item() - 3, prior_lowdB[num, 0, i].item() + 3, steps=91)
        aodr = torch.linspace(prior_lowdB[num, 1, i].item() - 3, prior_lowdB[num, 1, i].item() + 3, steps=91)
        aoar = torch.linspace(prior_lowdB[num, 2, i].item() - 3, prior_lowdB[num, 2, i].item() + 3, steps=91)
        posterior = prior_lowdB[num, :, i]

        p_ar = 0
        p_ad = 0
        p_aa = 0
        for i_ar in range(len(ar)):
            if ar[i_ar] < 0:
                ar[i_ar] = 0

            for i_ad in range(len(aodr)):
                #if aodr[i_ad] < -torch.pi:
                #    aodr[i_ad] = -torch.pi
                #if aodr[i_ad] > torch.pi:
                #    aodr[i_ad] = torch.pi

                for i_aa in range(len(aoar)):
                    #if aoar[i_aa] < -torch.pi:
                    #    aoar[i_aa] = -torch.pi
                    #if aoar[i_aa] > torch.pi:
                    #    aoar[i_aa] = torch.pi

                    h_test = ((ar[i_ar]) ** 2) * (
                                torch.abs(torch.dot(torch.conj(array_response((x_true[2, i-1] / 180 * torch.pi), Nr)),
                                                    array_response((aoar[i_aa] / 180 * torch.pi), Nr))) ** 2) \
                            * (torch.abs(torch.dot(torch.conj(array_response((aodr[i_ad] / 180 * torch.pi), Nt)),
                                           array_response(((x_true[1, i-1] / 180 * torch.pi)), Nt))) ** 2)
                    loss = torch.abs(training_input_no_off[num, j, i] - h_test) ** 2
                    if loss < tres:
                        posterior = torch.tensor([ar[i_ar], aodr[i_ad], aoar[i_aa]])
                        tres = loss
                        p_ar = i_ar
                        p_ad = i_ad
                        p_aa = i_aa
                    #print(j, i, i_ar, i_ad, i_aa)

        #target[1] = target[1] * 180 / torch.pi
        #posterior[1] = posterior[1] * 180 / torch.pi
        #target[2] = target[2] * 180 / torch.pi
        #posterior[2] = posterior[2] * 180 / torch.pi
        pos_low[num, j, :, i] = posterior

        print("i_ar:", p_ar, "i_ad:", p_ad, "i_aa:", p_aa)
        print("i: ", i, "prior low: ", torch.tensor([prior_lowdB[num, 0, i], prior_lowdB[num, 1, i], prior_lowdB[num, 2, i]]))
        # print("i: ", i, "ground truth low1: ", training_target[i, ::])
        print("i: ", i, "ground truth low1: ", torch.tensor([training_target[num, 0, i], training_target[num, 1, i], training_target[num, 2, i]]))
        print("i: ", i, "ground truth low2: ", target)
        print("i: ", i, "estimated: ", posterior)
        print("i: ", i, "loss: ", tres)
        all_MSE_gs_low[j] += torch.norm(target - posterior) ** 2
        all_loss_gs_low[j] += tres
        MSE_gs_low[0, j] += torch.abs(target[0] - posterior[0]) ** 2
        MSE_gs_low[1, j] += torch.abs(target[1] - posterior[1]) ** 2
        MSE_gs_low[2, j] += torch.abs(target[2] - posterior[2]) ** 2

    MSE_gs_low[:, j] = MSE_gs_low[:, j] / iter
    all_MSE_gs_low[j] = all_MSE_gs_low[j] / iter
    all_loss_gs_low[j] = all_loss_gs_low[j] / iter



# grid search mid snr prior
for j in range(4):
    pos_mid[num, j, :, 0] = x_true[:, 0]
    for i in range(1, ind_length):
        target = torch.tensor([training_target[num, 0, i], training_target[num, 1, i], training_target[num, 2, i]])

        h_est = ((prior_middB[num, 0, i]) ** 2) * (
                torch.abs(torch.dot(torch.conj(array_response((x_true[2, i - 1] / 180 * torch.pi), Nr)),
                                    array_response((prior_middB[num, 2, i] / 180 * torch.pi), Nr))) ** 2) \
                * (torch.abs(torch.dot(torch.conj(array_response((prior_middB[num, 1, i] / 180 * torch.pi), Nt)),
                                       array_response((x_true[1, i - 1] / 180 * torch.pi), Nt))) ** 2)
        
        tres = torch.abs(training_input_no_off[num, j, i] - h_est) ** 2
        # tres = 10000
        ar = torch.linspace(prior_middB[num, 0, i].item() - 1.5, prior_middB[num, 0, i].item() + 1.5, steps=61)
        aodr = torch.linspace(prior_middB[num, 1, i].item() - 1.5, prior_middB[num, 1, i].item() + 1.5, steps=61)
        aoar = torch.linspace(prior_middB[num, 2, i].item() - 1.5, prior_middB[num, 2, i].item() + 1.5, steps=61)
        posterior = prior_middB[num, :, i]

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
                            torch.abs(torch.dot(
                                torch.conj(array_response((x_true[2, i - 1] / 180 * torch.pi), Nr)),
                                array_response((aoar[i_aa] / 180 * torch.pi), Nr))) ** 2) \
                             * (torch.abs(torch.dot(torch.conj(array_response((aodr[i_ad] / 180 * torch.pi), Nt)),
                                                    array_response(
                                                        ((x_true[1, i - 1] / 180 * torch.pi)),
                                                        Nt))) ** 2)
                    loss = torch.abs(training_input_no_off[num, j, i] - h_test) ** 2
                    if loss < tres:
                        posterior = torch.tensor([ar[i_ar], aodr[i_ad], aoar[i_aa]])
                        tres = loss
                        p_ar = i_ar
                        p_ad = i_ad
                        p_aa = i_aa
                    # print(j, i, i_ar, i_ad, i_aa)

        # target[1] = target[1] * 180 / torch.pi
        # posterior[1] = posterior[1] * 180 / torch.pi
        # target[2] = target[2] * 180 / torch.pi
        # posterior[2] = posterior[2] * 180 / torch.pi
        pos_mid[num, j, :, i] = posterior

        print("i_ar:", p_ar, "i_ad:", p_ad, "i_aa:", p_aa)
        print("i: ", i, "prior mid: ",
              torch.tensor([prior_middB[num, 0, i], prior_middB[num, 1, i], prior_middB[num, 2, i]]))
        # print("i: ", i, "ground truth mid1: ", training_target[i, ::])
        print("i: ", i, "ground truth mid1: ",
              torch.tensor([training_target[num, 0, i], training_target[num, 1, i], training_target[num, 2, i]]))
        print("i: ", i, "ground truth mid2: ", target)
        print("i: ", i, "estimated: ", posterior)
        print("i: ", i, "loss: ", tres)
        all_MSE_gs_mid[j] += torch.norm(target - posterior) ** 2
        all_loss_gs_mid[j] += tres
        MSE_gs_mid[0, j] += torch.abs(target[0] - posterior[0]) ** 2
        MSE_gs_mid[1, j] += torch.abs(target[1] - posterior[1]) ** 2
        MSE_gs_mid[2, j] += torch.abs(target[2] - posterior[2]) ** 2

    MSE_gs_mid[:, j] = MSE_gs_mid[:, j] / iter
    all_MSE_gs_mid[j] = all_MSE_gs_mid[j] / iter
    all_loss_gs_mid[j] = all_loss_gs_mid[j] / iter


# grid search high snr prior
for j in range(4):
    pos_high[num, j, :, 0] = x_true[:, 0]
    for i in range(1, ind_length):
        target = torch.tensor([training_target[num, 0, i], training_target[num, 1, i], training_target[num, 2, i]])

        h_est = ((prior_highdB[num, 0, i]) ** 2) * (
                torch.abs(torch.dot(torch.conj(array_response((x_true[2, i - 1] / 180 * torch.pi), Nr)),
                                    array_response((prior_highdB[num, 2, i] / 180 * torch.pi), Nr))) ** 2) \
                * (torch.abs(torch.dot(torch.conj(array_response((prior_highdB[num, 1, i] / 180 * torch.pi), Nt)),
                                       array_response((x_true[1, i - 1] / 180 * torch.pi), Nt))) ** 2)
        
        tres = torch.abs(training_input_no_off[num, j, i] - h_est) ** 2
        # tres = 10000
        ar = torch.linspace(prior_highdB[num, 0, i].item() - 0.75, prior_highdB[num, 0, i].item() + 0.75, steps=31)
        aodr = torch.linspace(prior_highdB[num, 1, i].item() - 0.75, prior_highdB[num, 1, i].item() + 0.75, steps=31)
        aoar = torch.linspace(prior_highdB[num, 2, i].item() - 0.75, prior_highdB[num, 2, i].item() + 0.75, steps=31)
        posterior = prior_highdB[num, :, i]

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
                            torch.abs(torch.dot(
                                torch.conj(array_response((x_true[2, i - 1] / 180 * torch.pi), Nr)),
                                array_response((aoar[i_aa] / 180 * torch.pi), Nr))) ** 2) \
                             * (torch.abs(torch.dot(torch.conj(array_response((aodr[i_ad] / 180 * torch.pi), Nt)),
                                                    array_response(
                                                        ((x_true[1, i - 1] / 180 * torch.pi)),
                                                        Nt))) ** 2)
                    loss = torch.abs(training_input_no_off[num, j, i] - h_test) ** 2
                    if loss < tres:
                        posterior = torch.tensor([ar[i_ar], aodr[i_ad], aoar[i_aa]])
                        tres = loss
                        p_ar = i_ar
                        p_ad = i_ad
                        p_aa = i_aa
                    # print(j, i, i_ar, i_ad, i_aa)

        # target[1] = target[1] * 180 / torch.pi
        # posterior[1] = posterior[1] * 180 / torch.pi
        # target[2] = target[2] * 180 / torch.pi
        # posterior[2] = posterior[2] * 180 / torch.pi
        pos_high[num, j, :, i] = posterior

        print("i_ar:", p_ar, "i_ad:", p_ad, "i_aa:", p_aa)
        print("i: ", i, "prior high: ",
              torch.tensor([prior_highdB[num, 0, i], prior_highdB[num, 1, i], prior_highdB[num, 2, i]]))
        # print("i: ", i, "ground truth high1: ", training_target[i, ::])
        print("i: ", i, "ground truth high1: ",
              torch.tensor([training_target[num, 0, i], training_target[num, 1, i], training_target[num, 2, i]]))
        print("i: ", i, "ground truth high2: ", target)
        print("i: ", i, "estimated: ", posterior)
        print("i: ", i, "loss: ", tres)
        all_MSE_gs_high[j] += torch.norm(target - posterior) ** 2
        all_loss_gs_high[j] += tres
        MSE_gs_high[0, j] += torch.abs(target[0] - posterior[0]) ** 2
        MSE_gs_high[1, j] += torch.abs(target[1] - posterior[1]) ** 2
        MSE_gs_high[2, j] += torch.abs(target[2] - posterior[2]) ** 2

    MSE_gs_high[:, j] = MSE_gs_high[:, j] / iter
    all_MSE_gs_high[j] = all_MSE_gs_high[j] / iter
    all_loss_gs_high[j] = all_loss_gs_high[j] / iter


RMSE_gs_low = MSE_gs_low ** 0.5
RMSE_gs_mid = MSE_gs_mid ** 0.5
RMSE_gs_high = MSE_gs_high ** 0.5

all_RMSE_gs_low = all_MSE_gs_low ** 0.5
all_RMSE_gs_mid = all_MSE_gs_mid ** 0.5
all_RMSE_gs_high = all_MSE_gs_high ** 0.5

MSE_gs_low = 10 * torch.log10(MSE_gs_low)
MSE_gs_mid = 10 * torch.log10(MSE_gs_mid)
MSE_gs_high = 10 * torch.log10(MSE_gs_high)

all_MSE_gs_low = 10 * torch.log10(all_MSE_gs_low)
all_MSE_gs_mid = 10 * torch.log10(all_MSE_gs_mid)
all_MSE_gs_high = 10 * torch.log10(all_MSE_gs_high)

print("grid search, 1/r^2 = -10dB, MSE:", MSE_gs_low)
print("grid search, 1/r^2 = -15dB, MSE:", MSE_gs_mid)
print("grid search, 1/r^2 = -20dB, MSE:", MSE_gs_high)
print(all_MSE_gs_low)
print(all_MSE_gs_mid)
print(all_MSE_gs_high)

print("grid search, 1/r^2 = -10dB, RMSE:", RMSE_gs_low)
print("grid search, 1/r^2 = -15dB, RMSE:", RMSE_gs_mid)
print("grid search, 1/r^2 = -20dB, RMSE:", RMSE_gs_high)
print(all_RMSE_gs_low)
print(all_RMSE_gs_mid)
print(all_RMSE_gs_high)


print(all_loss_gs_low)
print(all_loss_gs_mid)
print(all_loss_gs_high)

