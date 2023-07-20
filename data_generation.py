import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import pyplot as plt

# from autograd import grad, jocobian
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # pi


# array response vector
def array_response(x, N):
    a = torch.empty(N, dtype=torch.complex64)
    for i in range(N):
        a[i] = torch.cos(torch.pi * torch.sin(x) * i) + 1j * torch.sin(torch.pi * torch.sin(x) * i)

    return (1 / N ** 0.5) * a


# parameters setting
Nr = 4
Nt = 4
SNR = torch.tensor([-10, 0, 10, 20])
n_PL = 2
sigma_alpha = (Nr * Nt) ** 0.5
sigma_obs_n = (1 * sigma_alpha) * (10 ** (SNR / -20))
Ns = 10

prior_q_p = torch.tensor([2, 1, 0.5])
prior_q_a = torch.tensor([2, 1, 0.5])

sigma_r = 1
sigma_omega = 1 # * torch.pi / 180 degree

y_UE = torch.ones(101) * 50  # y coordinate
x_UE = torch.linspace(0, 10, steps=101)  # x coordinate
omega_UE = torch.linspace(0, 0.1, steps=101)  # attitude

ind_length = 100 + 1
n_it = 50


r_x = torch.zeros(size=[n_it, 1, ind_length])
r_y = torch.zeros(size=[n_it, 1, ind_length])
r_omega = torch.zeros(size=[n_it, 1, ind_length])

training_target = torch.zeros(size=[n_it, 3, ind_length])  # x_k ground truth [pl, AoD, AoA]
training_input = torch.zeros(size=[n_it, 4, ind_length])  # |y_k|^2, 4 SNR
training_input_no_off = torch.zeros(size=[n_it, 4, ind_length])  # |y_k|^2 and h_k without offset
sensor_target = torch.zeros(size=[n_it, 3, ind_length])  # x_k tracked by spatial sensor

prior_lowdB = torch.zeros(size=[n_it, 3, ind_length])  # synthetic prior
prior_middB = torch.zeros(size=[n_it, 3, ind_length])
prior_highdB = torch.zeros(size=[n_it, 3, ind_length])

miu_k = torch.zeros(size=[n_it, 3, ind_length])  # miu_k
A_k = torch.zeros(size=[n_it, 3, 3, ind_length])  # A_k
Q_k = torch.zeros(size=[n_it, 3, 3, ind_length])  # process noise
R_k = torch.zeros(size=[n_it, 4, ind_length])  # obs noise, 4 SNR value
R_k_no_off = torch.zeros(size=[n_it, 4, ind_length])  # obs noise with h_no_off


for j_it in range(n_it):
    y_UE_e = y_UE + sigma_r * torch.randn(101)  # r_x
    x_UE_e = x_UE + sigma_r * torch.randn(101)  # r_y
    omega_UE_e = torch.zeros_like(omega_UE)
    omega_error = torch.zeros_like(omega_UE)  # r_omega

    for i in range(1, ind_length):
        omega_error[i] = sigma_omega * torch.randn(1)
        omega_UE_e[i] = omega_UE_e[i - 1] + ((0.02 * torch.pi / 180) + omega_error[i]) * 0.05

    r_x[j_it, ::] = x_UE_e
    r_y[j_it, ::] = y_UE_e

    r_omega[j_it, ::] = omega_error

    # sigma_alpha = (Nr * Nt) ** 0.5
    alpha_0 = (0.5 * (sigma_alpha ** 2)) ** 0.5 * (torch.randn(1) + 1j * torch.randn(1))

    x_true = torch.zeros(3, 101)
    x_true[0, 0] = torch.abs(alpha_0)
    x_true[1, 0] = torch.tensor(0.0)
    x_true[2, 0] = torch.tensor(180.0) # torch.pi

    for i_true in range(1, ind_length):
        rho = ((x_UE[i_true - 1] ** 2 + y_UE[i_true - 1] ** 2) ** 0.5 / (x_UE[i_true] ** 2 + y_UE[i_true] ** 2) ** 0.5) ** n_PL
        A = torch.tensor([[rho, 0, 0], [0, 1, 0], [0, 0, 1]])
        g = (torch.atan(x_UE[i_true] / y_UE[i_true]) - torch.atan(x_UE[i_true - 1] / y_UE[i_true - 1])) * 180 / torch.pi
        delta = 1 / (x_UE[i_true - 1] ** 2 + y_UE[i_true - 1] ** 2) + 1 / (x_UE[i_true] ** 2 + y_UE[i_true] ** 2)

        x_true[:, i_true] = torch.matmul(A, x_true[:, i_true - 1]) + torch.tensor([0, g, g + (0.02 * 180 / torch.pi) * 0.05])

    x_real = torch.zeros(3, 101)
    x_real[0, 0] = torch.abs(alpha_0) #+ 0.05 * torch.randn(1)
    x_real[1, 0] = torch.tensor(0.0) #+ 0.05 * torch.randn(1)
    x_real[2, 0] = torch.tensor(180.0) # torch.pi #+ 0.05 * torch.randn(1)
    training_target[j_it, :, 0] = x_real[:, 0]

    x_sensor = torch.zeros(3, 101)
    x_sensor[0, 0] = torch.abs(alpha_0)  + 0.05 * torch.randn(1)
    x_sensor[1, 0] = torch.tensor(0.0)  + 0.05*180/torch.pi * torch.randn(1)
    x_sensor[2, 0] = torch.tensor(180.0)  + 0.05*180/torch.pi * torch.randn(1)

    for j in range(1, ind_length):
        rho_e = ((x_UE_e[j - 1] ** 2 + y_UE_e[j - 1] ** 2) ** 0.5 / (x_UE_e[j] ** 2 + y_UE_e[j] ** 2) ** 0.5) ** n_PL
        A_e = torch.tensor([[rho_e, 0, 0], [0, 1, 0], [0, 0, 1]])
        g_e = (torch.atan(x_UE_e[j] / y_UE_e[j]) - torch.atan(x_UE_e[j - 1] / y_UE_e[j - 1])) * 180 / torch.pi
        miu_e = torch.tensor([0, g_e, g_e + ((0.02 * 180 / torch.pi) + omega_error[j]) * 0.05])

        delta_e = 1 / (x_UE_e[j - 1] ** 2 + y_UE_e[j - 1] ** 2) + 1 / (x_UE_e[j] ** 2 + y_UE_e[j] ** 2)

        x_real[:, j] = torch.matmul(A_e, x_true[:, j - 1]) + miu_e
        x_sensor[:, j] = torch.matmul(A_e, x_sensor[:, j - 1]) + miu_e

        Q = torch.tensor([[((n_PL * x_real[0, j - 1] * rho_e) ** 2) * delta_e, 0, 0],
                          [0, delta_e * 180 / torch.pi, delta_e * 180 / torch.pi],
                          [0, delta_e * 180 / torch.pi, delta_e * 180 / torch.pi + (0.05 * sigma_omega) ** 2]])

        print(Q)

        n = MultivariateNormal(torch.zeros(3), Q)
        n_s = n.sample()

        print(n_s)

        x_real[:, j] += n_s
        x_sensor[:, j] += n_s

        A_k[j_it, :, :, j] = A_e
        miu_k[j_it, :, j] = miu_e
        Q_k[j_it, :, :, j] = Q

        training_target[j_it, :, j] = x_real[:, j]
        sensor_target[j_it, :, j] = x_sensor[:, j]

        prior_lowdB[j_it, 0, j] = training_target[j_it, 0, j] + prior_q_p[0] * torch.randn(1)
        prior_lowdB[j_it, 1, j] = training_target[j_it, 1, j] + (prior_q_a[0] ) * torch.randn(1)
        prior_lowdB[j_it, 2, j] = training_target[j_it, 2, j] + (prior_q_a[0] ) * torch.randn(1)

        prior_middB[j_it, 0, j] = training_target[j_it, 0, j] + prior_q_p[1] * torch.randn(1)
        prior_middB[j_it, 1, j] = training_target[j_it, 1, j] + (prior_q_a[1] ) * torch.randn(1)
        prior_middB[j_it, 2, j] = training_target[j_it, 2, j] + (prior_q_a[1] ) * torch.randn(1)

        prior_highdB[j_it, 0, j] = training_target[j_it, 0, j] + prior_q_p[2] * torch.randn(1)
        prior_highdB[j_it, 1, j] = training_target[j_it, 1, j] + (prior_q_a[2] ) * torch.randn(1)
        prior_highdB[j_it, 2, j] = training_target[j_it, 2, j] + (prior_q_a[2] ) * torch.randn(1)

    for j in range(1, ind_length):
        h = (x_real[0, j] ** 2) * (torch.abs(torch.dot(torch.conj(array_response((x_true[2, j - 1] * torch.pi / 180) + torch.pi / 18, Nr)),
                                                       array_response((x_real[2, j] * torch.pi / 180), Nr))) ** 2) \
            * (torch.abs(torch.dot(torch.conj(array_response((x_real[1, j] * torch.pi / 180), Nt)),
                                   array_response((x_true[1, j - 1] * torch.pi / 180) + torch.pi / 18, Nt))) ** 2)

        h1 = (x_real[0, j] ** 2) * (torch.abs(torch.dot(torch.conj(array_response((x_true[2, j - 1] * torch.pi / 180), Nr)),
                                                       array_response((x_real[2, j] * torch.pi / 180), Nr))) ** 2) \
            * (torch.abs(torch.dot(torch.conj(array_response((x_real[1, j] * torch.pi / 180), Nt)),
                                   array_response((x_true[1, j - 1] * torch.pi / 180), Nt))) ** 2)

        for j_noise in range(4):
            v = 4 * h * (sigma_obs_n[j_noise] ** 2)
            v1 = 4 * h1 * (sigma_obs_n[j_noise] ** 2)

            y = h + (v ** 0.5) * torch.rand(1)
            y1 = h1 + (v1 ** 0.5) * torch.rand(1)

            training_input[j_it, j_noise, j] = y
            training_input_no_off[j_it, j_noise, j] = y1

            R_k[j_it, j_noise, j] = v
            R_k_no_off[j_it, j_noise, j] = v1


torch.save([x_true, sensor_target, training_target, training_input, training_input_no_off, prior_lowdB, prior_middB, prior_highdB, miu_k, A_k, Q_k, R_k, R_k_no_off, r_x, r_y, r_omega], 'test_data.pt')


# [x_true, sensor_target, training_target, training_input, training_input_no_off, prior_lowdB, prior_middB, prior_highdB, miu_k, A_k, Q_k, R_k, R_k_no_off, r_x, r_y, r_omega] = torch.load('final_data12.0.pt')


