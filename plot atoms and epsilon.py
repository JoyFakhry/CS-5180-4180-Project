import numpy as np
import matplotlib.pyplot as plt


def rolling_average(data, *, window_size=10):
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]


atoms_trial_1 = np.load('Data/Atoms/64 atoms trial 3 200 episodes.npy')
atoms_trial_2 = np.load('Data/Atoms/64 atoms trial 2 200 episodes.npy')
atoms_trial_3 = np.load('Data/Atoms/16 atoms trial 1 200 episodes.npy')
atoms_trial_4 = np.load('Data/Atoms/64 atoms trial 4 200 episodes.npy')
atoms_trial_5 = np.load('Data/Atoms/64 atoms trial 5 200 episodes.npy')


# Data = [Atoms, Trial, Episode]
# print(atoms_trial_1.shape)
# print(atoms_trial_2.shape)
# print(atoms_trial_3.shape)
atoms = [2, 4, 8, 16, 32, 64]

trials = [atoms_trial_1, atoms_trial_2, atoms_trial_3, atoms_trial_4, atoms_trial_5]
atoms_2 = np.zeros((len(trials), 200))
for index, trial in enumerate(trials):
    atoms_2[index] = trial[0]
avg = atoms_2.mean(axis=0)
std = atoms_2.std(axis=0)
length = len(avg)
y_err = 1.96 * std * np.sqrt(1 / length)
plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)
plt.plot(rolling_average(avg), label='2 atoms')

atoms_4 = np.zeros((len(trials), 200))
for index, trial in enumerate(trials):
    atoms_4[index] = trial[1]
avg = atoms_4.mean(axis=0)
std = atoms_4.std(axis=0)
length = len(avg)
y_err = 1.96 * std * np.sqrt(1 / length)
plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)
plt.plot(rolling_average(avg), label='4 atoms')

atoms_8 = np.zeros((len(trials), 200))
for index, trial in enumerate(trials):
    atoms_8[index] = trial[2]
avg = atoms_8.mean(axis=0)
std = atoms_8.std(axis=0)
length = len(avg)
y_err = 1.96 * std * np.sqrt(1 / length)
plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)
plt.plot(rolling_average(avg), label='8 atoms')

atoms_16 = np.zeros((len(trials), 200))
for index, trial in enumerate(trials):
    atoms_16[index] = trial[3]
avg = atoms_16.mean(axis=0)
std = atoms_16.std(axis=0)
length = len(avg)
y_err = 1.96 * std * np.sqrt(1 / length)
plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)
plt.plot(rolling_average(avg), label='16 atoms')

atoms_32 = np.zeros((len(trials), 200))
for index, trial in enumerate(trials):
    atoms_32[index] = trial[4]
atoms_32 = atoms_32[~np.all(atoms_32 == 0, axis=1)]
avg = atoms_32.mean(axis=0)
std = atoms_32.std(axis=0)
length = len(avg)
y_err = 1.96 * std * np.sqrt(1 / length)
plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)
plt.plot(rolling_average(avg), label='32 atoms')

atoms_64 = np.zeros((len(trials), 200))
for index, trial in enumerate(trials):
    atoms_64[index] = trial[5]
atoms_64 = atoms_64[~np.all(atoms_64 == 0, axis=1)]
avg = atoms_64.mean(axis=0)
std = atoms_64.std(axis=0)
length = len(avg)
y_err = 1.96 * std * np.sqrt(1 / length)
plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)
plt.plot(rolling_average(avg), label='64 atoms')


plt.legend()
plt.title('Varying number of atoms on Stochastic Environment')
plt.xlabel('Episodes')
plt.ylabel('Number of steps')
plt.savefig('Pics/Varying atoms.png')
# plt.show()
plt.close()



epsilon_trial_1 = np.load('Data/Epsilon/0.5 epsilon trial 1 200 episodes.npy')
epsilon_trial_2 = np.load('Data/Epsilon/0.5 epsilon trial 2 200 episodes.npy')
epsilon_trial_3 = np.load('Data/Epsilon/0.5 epsilon trial 3 200 episodes.npy')
epsilon_trial_4 = np.load('Data/Epsilon/0.5 epsilon trial 4 200 episodes.npy')
epsilon_trial_5 = np.load('Data/Epsilon/0.5 epsilon trial 5 200 episodes.npy')


# Data = [Epsilon, Trial, Episode]
print(epsilon_trial_1.shape)
print(epsilon_trial_2.shape)
print(epsilon_trial_3.shape)
trials = [epsilon_trial_1, epsilon_trial_2, epsilon_trial_3, epsilon_trial_4, epsilon_trial_5]
eps = [0.1, 0.2, 0.3, 0.4, 0.5]

# eps_0 = np.zeros((len(eps), 200))
# for index, trial in enumerate(trials):
#     eps_0[index] = trial[0]
# avg = eps_0.mean(axis=0)
# std = eps_0.std(axis=0)
# length = len(avg)
# y_err = 1.96 * std * np.sqrt(1 / length)
# plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)
# plt.plot(rolling_average(avg), label='e=0')

eps_1 = np.zeros((len(eps), 200))
for index, trial in enumerate(trials):
    eps_1[index] = trial[1]
avg = eps_1.mean(axis=0)
std = eps_1.std(axis=0)
length = len(avg)
y_err = 1.96 * std * np.sqrt(1 / length)
plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)
plt.plot(rolling_average(avg), label=r'$\varepsilon$ = 0.1')

eps_2 = np.zeros((len(eps), 200))
for index, trial in enumerate(trials):
    eps_2[index] = trial[2]
avg = eps_2.mean(axis=0)
std = eps_2.std(axis=0)
length = len(avg)
y_err = 1.96 * std * np.sqrt(1 / length)
plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)
plt.plot(rolling_average(avg), label=r'$\varepsilon$ = 0.2')

eps_3 = np.zeros((len(eps), 200))
for index, trial in enumerate(trials):
    eps_3[index] = trial[3]
avg = eps_3.mean(axis=0)
std = eps_3.std(axis=0)
length = len(avg)
y_err = 1.96 * std * np.sqrt(1 / length)
plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)
plt.plot(rolling_average(avg), label=r'$\varepsilon$ = 0.3')

eps_4 = np.zeros((len(eps), 200))
for index, trial in enumerate(trials):
    eps_4[index] = trial[4]
avg = eps_4.mean(axis=0)
std = eps_4.std(axis=0)
length = len(avg)
y_err = 1.96 * std * np.sqrt(1 / length)
plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)
plt.plot(rolling_average(avg), label=r'$\varepsilon$ = 0.4')

eps_5 = np.zeros((len(eps), 200))
for index, trial in enumerate(trials):
    eps_5[index] = trial[5]
avg = eps_5.mean(axis=0)
std = eps_5.std(axis=0)
length = len(avg)
y_err = 1.96 * std * np.sqrt(1 / length)
plt.fill_between(np.linspace(0, length - 1, length), avg - y_err, avg + y_err, alpha=0.2)
plt.plot(rolling_average(avg), label=r'$\varepsilon$ = 0.5')

plt.legend()
plt.title('Varying epsilon on Stochastic Environment')
plt.xlabel('Episodes')
plt.ylabel('Number of steps')
plt.savefig('Pics/Varying epsilon.png')
plt.show()

