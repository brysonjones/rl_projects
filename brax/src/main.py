import collections
from datetime import datetime
import functools
import math
import os
import time
import jax
from jax import numpy as jp
from typing import Any, Callable, Dict, Optional, Sequence

import brax_image.image as image
import brax
import matplotlib.pyplot as plt

from brax import envs
from brax.io import metrics
from brax.training.agents.ppo import train as ppo
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.wrappers import torch as torch_wrapper
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

env_name = 'ant'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
backend = 'positional'  # @param ['generalized', 'positional', 'spring']

env = envs.get_environment(env_name=env_name,
                           backend=backend)
state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))
im = image.render_array(env.sys, state.pipeline_state, 320, 320)

plt.imshow(im)
plt.show()