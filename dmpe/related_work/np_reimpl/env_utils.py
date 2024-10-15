import numpy as np


def simulate_ahead_with_env(env, obs, state, actions):
    observations = []
    observations.append(obs)

    print(actions.shape)
    print(actions[0, :][None, :].shape)

    for i in range(actions.shape[0]):
        obs, state = env.vmap_step(state, actions[i, :][None, :])
        observations.append(obs)

    return np.vstack(observations), state
