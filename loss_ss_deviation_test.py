import torch
import torch.nn as nn
import numpy as np
from prettytable import PrettyTable
from matplotlib import pyplot as plt
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
prior_q_p = torch.tensor([5, 3, 1])
prior_q_a = torch.tensor([10, 5, 1])

# [ground_truth, prior_lowdB, prior_highdB, noisy_obs, pure_h, pure_h_2] = torch.load('synthetic_data_offset_pure_-10_-20.pt')
###[ground_truth, prior_lowdB, prior_midd, prior_highdB, noisy_obs, obs_noise, pure_h, pure_h_2] = torch.load('synthetic_data_allone_10_2105_prior_wnoise_ybar.pt')

[x_true, sensor_target, training_target, training_input, training_input_no_off, prior_lowdB, prior_middB, prior_highdB, miu_k, A_k, Q_k, R_k, R_k_no_off, r_x, r_y, r_omega] = torch.load('test_data.pt')




loop_xb = 39
loop_sb = 40

dpl = torch.zeros(81)
dpl1 = torch.zeros(81)
dpl2 = torch.zeros(81)
dpl3 = torch.zeros(81)
dpl4 = torch.zeros(81)

daod = torch.zeros(81)
daod1 = torch.zeros(81)
daod2 = torch.zeros(81)
daod3 = torch.zeros(81)
daod4 = torch.zeros(81)

daoa = torch.zeros(81)

dpln = torch.zeros(81)
daodn = torch.zeros(81)
daoan = torch.zeros(81)

for j in range(loop_xb, loop_sb):

    # i = torch.randint(0, 1000, (1,))
    i = 0
    print("i: ", i)
    # print(pure_h[i, 0])
    # print(torch.abs(pure_h[i, 0]) ** 2)
    # print(pure_h_2[i, 0])

    def h_squ(t):
        return (t[0] ** 2) * (
                torch.abs(torch.dot(torch.conj(array_response((x_true[2, loop_xb]-3.0)* torch.pi/180, Nr)),
                                    array_response(t[2], Nr))) ** 2) \
               * (torch.abs(torch.dot(torch.conj(array_response(t[1] , Nt)),
                                      array_response((x_true[1, loop_xb]-3.0) * torch.pi/180, Nt))) ** 2)

    hx = h_squ(torch.tensor([training_target[i, 0, loop_sb], training_target[i, 1, loop_sb] * torch.pi/180, training_target[i, 2, loop_sb] * torch.pi/180]))
    print(torch.tensor([training_target[i, 0, loop_sb], training_target[i, 1, loop_sb] * torch.pi/180, training_target[i, 2, loop_sb] * torch.pi/180]))
    print(hx)
    print(training_input[i, 3, loop_sb])
    # print(hx - pure_h_2[i, 0])


    ar = torch.linspace(training_target[i, 0, loop_sb].item() - 1.5, training_target[i, 0, loop_sb].item() + 1.5,
                        steps=81)
    aodr = torch.linspace(training_target[i, 1, loop_sb].item() - 30,
                          training_target[i, 1, loop_sb].item() + 30,
                          steps=81)
    aoar = torch.linspace(training_target[i, 2, loop_sb].item() - 30,
                          training_target[i, 2, loop_sb].item() + 30,
                          steps=81)

    x_ar = ar - training_target[i, 0, loop_sb].item()
    x_aodr = aodr - training_target[i, 1, loop_sb].item()
    x_aoar = aoar - training_target[i, 2, loop_sb].item()

    aodr = aodr * torch.pi / 180
    aoar = aoar * torch.pi / 180



    for i_iter in range(len(ar)):
        x_pl = torch.tensor([ar[i_iter], training_target[i, 1, loop_sb]* torch.pi / 180, training_target[i, 2, loop_sb]* torch.pi / 180])
        x_pl1 = torch.tensor(
            [ar[i_iter], (training_target[i, 1, loop_sb] - 10)* torch.pi / 180, training_target[i, 2, loop_sb]* torch.pi / 180])
        x_pl2 = torch.tensor(
            [ar[i_iter], (training_target[i, 1, loop_sb] - 5)* torch.pi / 180, training_target[i, 2, loop_sb]* torch.pi / 180])
        x_pl3 = torch.tensor(
            [ar[i_iter], (training_target[i, 1, loop_sb] + 5)* torch.pi / 180, training_target[i, 2, loop_sb]* torch.pi / 180])
        x_pl4 = torch.tensor(
            [ar[i_iter], (training_target[i, 1, loop_sb] + 10)* torch.pi / 180, training_target[i, 2, loop_sb]* torch.pi / 180])


        x_aod = torch.tensor([training_target[i, 0, loop_sb], aodr[i_iter], training_target[i, 2, loop_sb]* torch.pi / 180])
        x_aod1 = torch.tensor([training_target[i, 0, loop_sb]-torch.tensor(1.5), aodr[i_iter], training_target[i, 2, loop_sb]* torch.pi / 180])
        x_aod2 = torch.tensor([training_target[i, 0, loop_sb]-torch.tensor(0.5), aodr[i_iter], training_target[i, 2, loop_sb]* torch.pi / 180])
        x_aod3 = torch.tensor([training_target[i, 0, loop_sb]+torch.tensor(0.5), aodr[i_iter], training_target[i, 2, loop_sb]* torch.pi / 180])
        x_aod4 = torch.tensor([training_target[i, 0, loop_sb]+torch.tensor(1.5), aodr[i_iter], training_target[i, 2, loop_sb]* torch.pi / 180])

        x_aoa = torch.tensor([training_target[i, 0, 0], training_target[i, 1, 0]* torch.pi / 180, aoar[i_iter]])

        hx_pl = h_squ(x_pl)
        hx_pl1 = h_squ(x_pl1)
        hx_pl2 = h_squ(x_pl2)
        hx_pl3 = h_squ(x_pl3)
        hx_pl4 = h_squ(x_pl4)

        hx_aod = h_squ(x_aod)
        hx_aod1 = h_squ(x_aod1)
        hx_aod2 = h_squ(x_aod2)
        hx_aod3 = h_squ(x_aod3)
        hx_aod4 = h_squ(x_aod4)

        hx_aoa = h_squ(x_aoa)

        pure_h_2 = hx

        dpl[i_iter] += torch.abs(pure_h_2 - hx_pl) ** 2
        dpl1[i_iter] += torch.abs(pure_h_2 - hx_pl1) ** 2
        dpl2[i_iter] += torch.abs(pure_h_2 - hx_pl2) ** 2
        dpl3[i_iter] += torch.abs(pure_h_2 - hx_pl3) ** 2
        dpl4[i_iter] += torch.abs(pure_h_2 - hx_pl4) ** 2


        daod[i_iter] += torch.abs(pure_h_2 - hx_aod) ** 2
        daod1[i_iter] += torch.abs(pure_h_2 - hx_aod1) ** 2
        daod2[i_iter] += torch.abs(pure_h_2 - hx_aod2) ** 2
        daod3[i_iter] += torch.abs(pure_h_2 - hx_aod3) ** 2
        daod4[i_iter] += torch.abs(pure_h_2 - hx_aod4) ** 2

        daoa[i_iter] += torch.abs(pure_h_2 - hx_aod) ** 2


dpl = dpl  #/(loop
dpl1 = dpl1  #/loop
dpl2 = dpl2  #/loop
dpl3 = dpl3  #/loop
dpl4 = dpl4  #/loop

daod = daod  #/loop
daod1 = daod1  #/loop
daod2 = daod2  #/loop
daod3 = daod3  #/loop
daod4 = daod4  #/loop

daoa = daoa  #/loop

dpln = dpl2  #/loop
daodn = daodn  #/loop
daoan = daoan  #/loop



zero_level = torch.zeros_like(x_ar)

# k = np.arange(0, 101, 1 / degree)
plt.xlabel('difference of the ground truth')
plt.ylabel('Value of loss function')

p1, = plt.plot(x_ar, dpl, 'b')
p2, = plt.plot(x_ar, zero_level, 'r--')
p3, = plt.plot(x_ar, dpl1, 'g--')
p4, = plt.plot(x_ar, dpl2, 'y-.')
p5, = plt.plot(x_ar, dpl3, 'k:')
p6, = plt.plot(x_ar, dpl4, 'c--')
plt.legend([p1, p3, p4, p5, p6], ['0°', '-10°', '-5°', '5°', '10°'])
'''
p1, = plt.plot(x_aodr, daod, 'b')
p2, = plt.plot(x_aodr, zero_level, 'r--')
p3, = plt.plot(x_aodr, daod1, 'g--')
p4, = plt.plot(x_aodr, daod2, 'y-.')
p5, = plt.plot(x_aodr, daod3, 'k:')
p6, = plt.plot(x_aodr, daod4, 'c--')
plt.legend([p1, p3, p4, p5, p6], ['0', '-1.5', '-0.5', '0.5', '1.5'])
'''
plt.title('Path Gain Norm')
#plt.title('Angle of departure')
#plt.ylim(-10, 200)
plt.grid(color='k', linestyle=':', linewidth=0.5)
plt.savefig('final_5_fig_k-1_pl.eps', dpi=300)
plt.show()

