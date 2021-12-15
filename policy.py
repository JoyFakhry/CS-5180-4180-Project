import numpy as np
from collections import defaultdict
from typing import Callable, Tuple, Sequence
import matplotlib.pyplot as plt

def argmax(arr: Sequence[float]) -> int:
    """Argmax that breaks ties randomly

    Takes in a list of values and returns the index of the item with the highest value, breaking ties randomly.

    Note: np.argmax returns the first index that matches the maximum, so we define this method to use in EpsilonGreedy and UCB agents.
    Args:
        arr: sequence of values
    """
    max_val = arr.max()
    argmax_bool = arr == max_val
    argmax_index = [idx for idx, isMax in enumerate(argmax_bool) if isMax]
    return np.random.choice(argmax_index)

def create_epsilon_policy(Q: defaultdict, epsilon: float) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = len(Q[0])

    def get_action(state: Tuple) -> int:
        # You can reuse code from ex1
        # Make sure to break ties arbitrarily
        if np.random.random() < epsilon:
            action = np.random.choice([i for i in range(num_actions)])
        else:
            action = argmax(Q[state])

        return action

    return get_action


# Midterm, Final, HWs, Project, Attendance
weights = [0.17, 0.3, 0.08, 0.4, 0.05]
grades = [0.9, 0.8, (100+100+95+100+90)/500, 0.8, .8]

weights = np.array(weights)
grades = np.array(grades)

grade = weights*grades
print(grade)
print(np.sum(grade))

bins_64 = np.load('Data/Epsilon/0.5 epsilon 1 trials 20 episodes.npy')
for i in bins_64:
    avg = i.mean(axis=0)
    plt.plot(avg)
plt.show()