import numpy as np
import json
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def main():
    n_trials = 5
    with open('savedData/t4_eps300_det_rewards.json', 'r') as f:
        data = np.array(json.load(f))
    with open('savedData/t4_eps300_sto_rewards.json', 'r') as f:
        s_data = np.array(json.load(f))
    v_data = data[0:n_trials,]
    v_data_mean = v_data.mean(axis=0)
    v_data_se = 1.96 * v_data.std(axis=0) / np.sqrt(n_trials)
    v_s_data = s_data[0:n_trials, ]
    v_s_data_mean = v_s_data.mean(axis=0)
    v_s_data_se = 1.96 * v_s_data.std(axis=0) / np.sqrt(n_trials)

    plt.figure()
    ax = plt.axes()
    X = [i for i in range(len(v_data_mean))]
    ax.plot(X, v_data_mean, label="det")
    ax.plot(X, v_s_data_mean, label="sto")
    ax.fill_between(X, v_data_mean+v_data_se, v_data_mean-v_data_se, alpha=0.2)
    ax.fill_between(X, v_s_data_mean + v_s_data_se, v_s_data_mean - v_s_data_se, alpha=0.2)
    ax.set_xlabel("episode")
    ax.set_ylabel("rewards")
    plt.title("distRL")
    plt.legend()
    plt.show()
    print(v_data_mean)


if __name__ == "__main__":
    main()
