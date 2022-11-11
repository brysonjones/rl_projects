
import numpy as np
import cv2

from network.memory import Memory

class MainModel():
    def __init__(self, network, optimizer):
        self._network = network
        self._optimizer = optimizer
        self._replay_memory = Memory()

    def get_action(current_obs):
        pass

    def preprocess(current_obs):
        pass

    def learn():
        pass

    def eval():
        pass