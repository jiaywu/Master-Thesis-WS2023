import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import pyplot as plt
#from autograd import grad, jocobian
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # pi


def array_response(x, N):
    a = torch.empty(N, dtype=torch.complex128)
    for i in range(N):
        a[i] = torch.cos(torch.pi * torch.sin(x) * i) + 1j * torch.sin(torch.pi * torch.sin(x) * i)

    return (1 / N ** 0.5) * a
# print(array_response(torch.tensor(torch.pi/6), 4))


LOSS1 = np.zeros(101)
LOSS2 = np.zeros(101)
LOSS3 = np.zeros(101)

Nr = 4
Nt = 4
#SNR = 10
n_PL = 2

y_UE = torch.ones(101) * 50  # y coordinate
x_UE = torch.linspace(0, 10, steps=101)  # x coordinate
omega_UE = torch.linspace(0, 0.1, steps=101)  # attitude

sigma_r = 1
sigma_omega = 1 # * torch.pi / 180

n_it = 50
n_it_low = 0
SNR_ind = 3  # SNR value used

[x_true, sensor_target, training_target, training_input, training_input_no_off, prior_lowdB, prior_middB, prior_highdB, miu_k, A_k, Q_k, R_k, R_k_no_off, r_x, r_y, r_omega] = torch.load('test_data.pt')


for j_it in range(n_it_low, n_it):

    x_real = torch.zeros(3, 101) # ground truth value
    x_real[0, 0] = training_target[j_it, 0, 0].squeeze()
    x_real[1, 0] = torch.tensor(0.0)
    x_real[2, 0] = torch.tensor(180.0)
    # x_real[3, 0] = torch.angle(alpha_0)

    x_faulty = torch.zeros(3, 101) # spatial sensor tested value
    x_faulty[0, 0] = training_target[j_it, 0, 0].squeeze() # + 0.05 * torch.randn(1)
    x_faulty[1, 0] = (torch.atan(r_x[j_it, 0, 0].squeeze() / r_y[j_it, 0, 0].squeeze())) * 180 / torch.pi # + 0.05 * torch.randn(1)
    x_faulty[2, 0] = (torch.atan(r_x[j_it, 0, 0].squeeze() / r_y[j_it, 0, 0].squeeze())) * 180 / torch.pi + torch.tensor(180.0)# + 0.05 * torch.randn(1)

    for j in range(1, 101):

        rho = ((x_UE[j - 1] ** 2 + y_UE[j - 1] ** 2) ** 0.5 / (x_UE[j] ** 2 + y_UE[j] ** 2) ** 0.5) ** n_PL

        x_real[0, j] = rho * x_real[0, j-1]
        x_real[1, j] = torch.atan(x_UE[j] / y_UE[j]) * 180 / torch.pi
        x_real[2, j] = torch.atan(x_UE[j] / y_UE[j]) * 180 / torch.pi + torch.tensor(180.0) + 0.05 * (0.02 * 180 / torch.pi) * j

        rho_e = A_k[j_it, 0, 0, j].squeeze()

        Sigma_x = Q_k[j_it, :, :, j].squeeze()
        n = MultivariateNormal(torch.zeros(3), Sigma_x)

        x_faulty[0, j] = sensor_target[j_it, 0, j].squeeze()
        x_faulty[1, j] = sensor_target[j_it, 1, j].squeeze()
        x_faulty[2, j] = sensor_target[j_it, 2, j].squeeze()

    x = torch.zeros(3, 101)
    x[0, 0] = training_target[j_it, 0, 0].squeeze()
    x[1, 0] = training_target[j_it, 1, 0].squeeze()
    x[2, 0] = training_target[j_it, 2, 0].squeeze()

    # print(torch.squeeze(x[:, 0]))

    m1x_posterior = x[:, 0].squeeze()
    m2x_posterior = torch.tensor([[0.05 ** 2, 0, 0], [0, 0.05 ** 2, 0], [0, 0, 0.05 ** 2]])
    m1x_prior = torch.empty_like(m1x_posterior)
    m2x_prior = torch.empty_like(m2x_posterior)


    for i_EKF in range(1, 101): # EKF loop

        A = A_k[j_it, :, :, i_EKF].squeeze()
        miu = miu_k[j_it, :, i_EKF].squeeze()
        Sigma_x = Q_k[j_it, :, :, i_EKF].squeeze()

        m1x_prior = torch.matmul(A, m1x_posterior) + miu
        # m1x_prior = m1x_prior + miu
        m2x_prior = torch.matmul(A, m2x_posterior)
        m2x_prior = torch.matmul(m2x_prior, torch.transpose(A, 0, 1)) + Sigma_x

        print("time index:", i_EKF)
        print("KG1: ", m2x_prior)
        print("noise_var:", R_k_no_off[j_it, SNR_ind, i_EKF].squeeze())

        # obs function
        def h_tilde(t):
            return (t[0] ** 2) * (torch.abs(torch.dot(
                torch.conj(array_response((m1x_posterior[2] * torch.pi / 180) + torch.pi / 18, Nr)),
                array_response((t[2] * torch.pi / 180), Nr))) ** 2) \
                   * (torch.abs(torch.dot(torch.conj(array_response((t[1] * torch.pi / 180), Nt)),
                                          array_response(
                                              (m1x_posterior[1] * torch.pi / 180) + torch.pi / 18,
                                              Nt))) ** 2)

        inputs = m1x_prior
        G = autograd.functional.jacobian(h_tilde, inputs)
        #print(G)
        KG_inter = torch.matmul(m2x_prior, G)
        KG_inter = torch.dot(G, KG_inter) + R_k[j_it, SNR_ind, i_EKF].squeeze()
        print("G:", G)
        print("KG2: ", KG_inter)
        # KG_inter = KG_inter ** -1
        KG = torch.matmul(m2x_prior, G) / KG_inter

        print("KGain:", KG)
        print("innovation:", training_input[j_it, SNR_ind, i_EKF].squeeze() - h_tilde(m1x_prior))
        print("product:", KG * (training_input[j_it, SNR_ind, i_EKF].squeeze() - h_tilde(m1x_prior)))
        m1x_posterior = m1x_prior + KG * (training_input[j_it, SNR_ind, i_EKF].squeeze() - h_tilde(m1x_prior)) #.to(torch.float)
        m2x_posterior = torch.matmul((torch.eye(3) - torch.outer(KG, G)), m2x_prior) #.float()

        x[:, i_EKF] = m1x_posterior
        #print(i_EKF)

    LOSS01 = (x[2, :] - x_real[2, :]).numpy()
    LOSS02 = (x_faulty[2, :] - x[2, :]).numpy()
    # LOSS03 = (x_faulty[2, :] - x_real[2, :]).numpy()
    LOSS1 += np.abs(LOSS01)
    LOSS2 += np.abs(LOSS02)
    # LOSS3 += np.abs(LOSS03)

# loss_fn = nn.MSELoss(reduction='mean')
LOSS1 = LOSS1 / (n_it - n_it_low)
LOSS2 = LOSS2 / (n_it - n_it_low)
# LOSS3 = LOSS3 / (n_it - n_it_low)

k = np.arange(0, 101, 1)
p1, = plt.plot(k[1:], np.abs(LOSS1[1:]), 'b')
p2, = plt.plot(k[1:], np.abs(LOSS2[1:]), 'r')
plt.xlabel('k')
plt.ylabel('|Error of AOD| / degrees')
plt.legend([p1, p2], ['EKF algorithm', 'sensor'])
# plt.savefig('img_test_2.eps', dpi=300)
plt.show()

for i in range(101):
    print("i: ", i)
    print("ture AoD", x_real[2, i])
    print("sensor AoD", x_faulty[2, i])
    print("EKF AoD", x[2, i])

print(np.mean(LOSS1))
print(np.mean(LOSS2))





