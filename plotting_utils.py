import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import jax.numpy as jnp

import exciting_environments as excenvs

from model_utils import simulate_ahead, simulate_ahead_with_env


def plot_sequence(observations, actions, tau, obs_labels, action_labels):
    """Plots a given sequence of observations and actions."""

    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 6),
        sharey=True
    )

    t = jnp.linspace(0, observations.shape[0]-1, observations.shape[0]) * tau

    for observation_idx in range(observations.shape[-1]):
        axs[0].plot(t, jnp.squeeze(observations[..., observation_idx]), label=obs_labels[observation_idx])

    axs[0].title.set_text("observations, timeseries")
    axs[0].legend()
    axs[0].set_xlabel(r"time $t$ in seconds")

    if observations.shape[-1] == 2:
        axs[1].scatter(jnp.squeeze(observations[..., 0]), jnp.squeeze(observations[..., 1]), s=1)
        axs[1].title.set_text("observations, together")

    for action_idx in range(actions.shape[-1]):
        axs[2].plot(t[:-1], jnp.squeeze(actions[..., action_idx]), label=action_labels[action_idx])

    axs[2].title.set_text("actions, timeseries")
    axs[2].legend()
    axs[2].set_xlabel(r"time $t$ in seconds")

    for ax in axs:
        ax.grid()

    fig.tight_layout()
    return fig, axs


def append_predictions_to_sequence_plot(
        fig,
        axs,
        starting_step,
        pred_observations,
        proposed_actions,
        tau,
        obs_labels,
        action_labels
):
    """Appends the future predictions to the given plot."""

    t = jnp.linspace(0, pred_observations.shape[0]-1, pred_observations.shape[0]) * tau
    t += tau * starting_step  # start where the trajectory left off

    colors = list(mcolors.CSS4_COLORS.values())[:pred_observations.shape[-1]]
    for observation_idx, color in zip(range(pred_observations.shape[-1]), colors):
        axs[0].plot(
            t,
            jnp.squeeze(pred_observations[..., observation_idx]),
            color=color,
            label="pred " + obs_labels[observation_idx]
        )

    if pred_observations.shape[-1] == 2:
        axs[1].scatter(
            jnp.squeeze(pred_observations[..., 0]),
            jnp.squeeze(pred_observations[..., 1]),
            s=1,
            color=mcolors.CSS4_COLORS["orange"]
        )

    colors = list(mcolors.CSS4_COLORS.values())[:pred_observations.shape[-1]]
    for action_idx, color in zip(range(proposed_actions.shape[-1]), colors):
        axs[2].plot(
            t[:-1],
            jnp.squeeze(proposed_actions[..., action_idx]),
            color=color,
            label="pred " + action_labels[action_idx]
        )

    return fig, axs


def plot_sequence_and_prediction(
        observations,
        actions,
        tau,
        obs_labels,
        actions_labels,
        model,
        init_obs,
        init_state,
        proposed_actions
):
    """Plots the current trajectory and appends the predictions from the optimization."""

    fig, axs = plot_sequence(
        observations=observations,
        actions=actions,
        tau=tau,
        obs_labels=obs_labels,
        action_labels=actions_labels,
    )

    if isinstance(model, excenvs.core_env.CoreEnvironment):
        pred_observations = simulate_ahead_with_env(
            env=model,
            init_obs=init_obs,
            init_state=init_state,
            actions=proposed_actions,
            env_state_normalizer=model.env_state_normalizer[0, :],
            action_normalizer=model.action_normalizer[0, :],
            static_params={key: value[0, :] for (key, value) in model.static_params.items()}
        )
    else:
        pred_observations = simulate_ahead(
            model=model,
            init_obs=init_obs,
            actions=proposed_actions,
            tau=tau
        )

    fig, axs = append_predictions_to_sequence_plot(
        fig=fig,
        axs=axs,
        starting_step=observations.shape[0],
        pred_observations=pred_observations,
        proposed_actions=proposed_actions,
        tau=tau,
        obs_labels=obs_labels,
        action_labels=actions_labels,
    )

    return fig, axs


def plot_2d_kde_as_contourf(
        p_est,
        x,
        observation_labels
):

    fig, axs = plt.subplots(
        figsize=(6, 6)
    )

    grid_len_per_dim = int(np.sqrt(x.shape[0]))
    x_plot = x.reshape((grid_len_per_dim, grid_len_per_dim, 2))

    cax = axs.contourf(
        x_plot[..., 0],
        x_plot[..., 1],
        p_est.reshape(x_plot.shape[:-1]),
        antialiased=False,
        levels=50,
        alpha=0.9,
        cmap=plt.cm.coolwarm
    )
    axs.set_xlabel(observation_labels[0])
    axs.set_ylabel(observation_labels[1])

    return fig, axs, cax


def plot_2d_kde_as_surface(
        p_est,
        x,
        observation_labels
):

    fig = plt.figure(figsize=(6, 6))
    axs = fig.add_subplot(111, projection='3d')

    grid_len_per_dim = int(np.sqrt(x.shape[0]))
    x_plot = x.reshape((grid_len_per_dim, grid_len_per_dim, 2))

    _ = axs.plot_surface(
        x_plot[..., 0],
        x_plot[..., 1],
        p_est.reshape(x_plot.shape[:-1]),
        antialiased=False,
        alpha=1,
        cmap=plt.cm.coolwarm
    )
    axs.set_xlabel(observation_labels[0])
    axs.set_ylabel(observation_labels[1])

    return fig, axs
