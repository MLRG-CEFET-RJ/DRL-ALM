import numpy as np
import pandas as pd
import gym
from gym import spaces
from scipy.stats import chi2

import torch
import torch.nn as nn


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        # self.pi = mlp(pi_sizes, activation, nn.Tanh) original entry
        self.pi = mlp(pi_sizes, activation, nn.Softmax) # Changed for ALMEnv
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

"""
ALM Environment
This environment is not part of the original OpenAI SpinningUp package
It's been included by the author
"""

class ALMEnv(gym.Env):
    """
    Custom Asset Liability Management environment, which follows gym interface
    Inputs are an asset value (scalar), a liability flow (numpy array of shape (T,))
    and a pandas DataFrame, with historical returns of available assets
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, T = 80, rate = .06, hist_returns = False):

        super(ALMEnv, self).__init__()

        self.asset = 10**6
        self.liability = chi2.pdf(np.linspace(0, 16, 101)[(101 - T):], 6)
        self.liab_PV = self.liability / (1 + rate) ** np.arange(1, T + 1)
        self.liability = self.liability * (self.asset / np.sum(self.liab_PV))

        if (hist_returns):
            self.historical_return = hist_returns
        else:
            self.historical_return = pd.DataFrame(np.array([[0.881818867, 1.277103375, 1.194665549, 1.196332479, 1.119897102, 1.143154236, 1.056897333],
                                                            [0.913401974, 1.329337917, 1.183150266, 1.152575668, 1.208069962, 1.283265184, 1.03141775],
                                                            [0.828484565, 1.436512041, 1.10733683, 1.119179339, 1.131582749, 1.190834926, 1.044573304],
                                                            [1.319369954, 0.587765708, 1.13880019, 1.123874437, 1.138172278, 1.075195418, 1.059023134],
                                                            [0.745057766, 1.826577896, 1.124799714, 1.09979594, 1.149761414, 1.235206438, 1.043120283],
                                                            [0.956926258, 1.010439144, 1.118628089, 1.097598994, 1.130256361, 1.218475311, 1.059090683],
                                                            [1.125795223, 0.818913771, 1.144601664, 1.116280628, 1.156939304, 1.144808206, 1.06503109],
                                                            [1.089401855, 1.073968355, 1.143073697, 1.085152406, 1.169810636, 1.342007027, 1.05838569],
                                                            [1.146366528, 0.845042, 1.025963782, 1.081912809, 1.027623167, 0.829212882, 1.059108181],
                                                            [1.133868351, 0.970877745, 1.113965671, 1.108091597, 1.116447326, 1.16609008, 1.064076166],
                                                            [1.470070025, 0.86685864, 1.071136115, 1.132591303, 1.154377104, 1.056908557, 1.10673498],
                                                            [0.834639418, 1.389351542, 1.233883065, 1.138430157, 1.15524236, 1.310909455, 1.062880551],
                                                            [1.015004142, 1.268567254, 1.152134718, 1.101916922, 1.12586988, 1.127526766, 1.029473499],
                                                            [1.171342201, 1.15032329, 1.107351925, 1.06420429, 1.098757474, 1.154167833, 1.037454821]]),
    columns = ['Cambio', 'Bovespa', 'IRF-M', 'IMA-S', 'IMA-B 5', 'IMA-B 5+', 'IPCA'],
    index = np.arange(2005, 2019))

        self.present_asset = self.asset
        self.present_liability = self.liability

        self.action_space = spaces.Box(low = 0, high = 1, shape = (self.historical_return.shape[1],), dtype = np.float32)
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = self.liability.shape, dtype = np.float32)

    def step(self, action):
        sim_ret = np.random.multivariate_normal(mean = self.historical_return.mean(axis = 0), cov = pd.DataFrame.cov(self.historical_return))
        self.present_asset = self.present_asset * np.sum(sim_ret * action) - self.present_liability[0]
        self.present_liability = np.append(self.present_liability[1:], 0) * sim_ret[0]

        terminal = False
        if self.present_asset < 0 or np.sum(self.present_liability) == 0:
            terminal = True

        if self.present_asset >= 0:
            reward = 1
        else:
            reward = 0

        observation = self.present_liability / self.present_asset

        info = None

        return observation, reward, terminal, info

    def reset(self):
        self.present_asset = self.asset
        self.present_liability = self.liability
        return(self.present_liability / self.present_asset)

    def render(self, mode = 'human', close = False):
        pass
