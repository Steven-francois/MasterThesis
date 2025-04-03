import numpy as np
import matplotlib.pyplot as plt

max_speed = 200

# prf1 = 0.12739207 *3.6
# prf2 = 0.12489419 *3.6
# prf3 = 0.1299919 *3.6
prf1 = 0.12489419 *3.6
prf2 = 0.1299919 *3.6
prf3 = 0.12739207 *3.6

nb_bins = 128

t1 = np.linspace(0, nb_bins*prf1, nb_bins) - nb_bins*prf1/2
t2 = np.linspace(0, nb_bins*prf2, nb_bins) - nb_bins*prf2/2
t3 = np.linspace(0, nb_bins*prf3, nb_bins) - nb_bins*prf3/2
t1_ = t1.copy()
t2_ = t2.copy()
t3_ = t3.copy()


target1 = np.zeros(nb_bins)
target2 = np.zeros(nb_bins)
target3 = np.zeros(nb_bins)

target2[:] = 0.2
target3[:] = 0.4

target1[nb_bins - 1] = 1
target2[nb_bins - 127] = 1
target3[nb_bins - 1] = 1
# target1[nb_bins - 117] = 1
# target2[nb_bins - 118] = 1
# target3[nb_bins - 116] = 1
# target1[nb_bins - 110] = 1
# target2[nb_bins - 117] = 1
# target3[nb_bins - 112] = 1
# target1[nb_bins - 91] = 1
# target2[nb_bins - 95] = 1
# target3[nb_bins - 93] = 1

plt.figure(figsize=(10, 6))
plt.title('Targets in Doppler Domain')
plt.plot(t1, target1, label='Target 1', marker='o')
plt.plot(t2, target2, label='Target 2', marker='o')
plt.plot(t3, target3, label='Target 3', marker='o')
for i in range(1, 6):
    plt.axvline(x=i*nb_bins*prf1-nb_bins*prf1/2, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=i*nb_bins*prf2-nb_bins*prf2/2, color='gray', linestyle='-.', alpha=0.5)
    plt.axvline(x=i*nb_bins*prf3-nb_bins*prf3/2, color='gray', linestyle='-', alpha=0.5)
    t1 += nb_bins*prf1
    t2 += nb_bins*prf2
    t3 += nb_bins*prf3
    target1 += 0.1
    target2 += 0.1
    target3 += 0.1
    plt.axvline(x=-i*nb_bins*prf1+nb_bins*prf1/2, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=-i*nb_bins*prf2+nb_bins*prf2/2, color='gray', linestyle='-.', alpha=0.5)
    plt.axvline(x=-i*nb_bins*prf3+nb_bins*prf3/2, color='gray', linestyle='-', alpha=0.5)
    plt.plot(t1, target1, label=f'Target 1 + {i} PRF', marker='o')
    plt.plot(t2, target2, label=f'Target 2 + {i} PRF', marker='o')
    plt.plot(t3, target3, label=f'Target 3 + {i} PRF', marker='o')
    t1_ -= nb_bins*prf1
    t2_ -= nb_bins*prf2
    t3_ -= nb_bins*prf3
    plt.plot(t1_, target1, label=f'Target 1 - {i} PRF', marker='o')
    plt.plot(t2_, target2, label=f'Target 2 - {i} PRF', marker='o')
    plt.plot(t3_, target3, label=f'Target 3 - {i} PRF', marker='o')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
plt.axhline(y=1, color='red', linestyle='--', alpha=0.5)

plt.xlabel('Doppler Bins')
plt.xlim(0, 6*nb_bins*prf1)
plt.ylabel('Amplitude')
# plt.legend()
plt.show()