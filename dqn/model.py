
import torch
import numpy as np
import random
import cv2

from network.memory import Memory

class MainModel():
    def __init__(self, network, optimizer, espilion=0.3, discount_gamma=0.99, replay_batch_size=16):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._network = network.to(self.device)
        self._optimizer = optimizer
        self._espilion = espilion
        self._discount_gamma = discount_gamma
        self._replay_batch_size = replay_batch_size
        self._replay_memory = Memory()

    def get_action(self, current_obs):
        if (random.random() < self._espilion):
            # explore
            return random.randint(0, self._network.action_size)
        else:
            # exploit
            q_values = self._network(current_obs)
            return q_values.cpu().data.numpy().argmax()

    def update_state(self, new_obs, prev_state=None, action=None, reward=None, done=None):
        '''
        prev_state: previous state that had already been preprocessed
        action: action take for the transition
        reward: reward received during the transition
        new_obs: observation after transition that has not yet been preprocessed
        '''
        processed_frame = self.preprocess(new_obs)

        # initialization step at the beginning of an episode
        if prev_state is None:
            # set the prev_state to be a repeating sequence of the new observation
            # TODO: should this be modular in length?
            new_processed_state = np.dstack((processed_frame, processed_frame, processed_frame, processed_frame))
            return np.expand_dims(np.transpose(new_processed_state, (2, 0, 1)), axis=0)  # don't need to add to replay memory at the beginning 
        else:
            # add and shift state frame with newly preprocessed observation data
            prev_state = prev_state.squeeze()
            processed_frame = np.expand_dims(processed_frame, axis=0)
            new_processed_state = np.concatenate((prev_state, processed_frame), axis=0)[1::]

        # store transition in replay memory
        self._replay_memory.add_transition(new_processed_state, prev_state, action, reward, done)
        return np.expand_dims(new_processed_state, axis=0)
    

    def preprocess(self, current_obs):
        # convert image to gray
        image_gray = cv2.cvtColor(current_obs, cv2.COLOR_RGB2GRAY)

        # determine square side length
        # height, width = image_gray.shape
        # slice = int((width-height)/2)
        # image_square = image_gray[:, slice:-slice]

        # resize
        image_scaled = cv2.resize(image_gray, (84, 84), interpolation=cv2.INTER_NEAREST) / 255

        return image_scaled

    def learn(self):
        prev_state_batch, action_batch, reward_batch, new_state_batch, done_batch = \
            self._replay_memory.get_samples(self._replay_batch_size)

        # arbitrarily check if any of the returned lists are empty
        if not prev_state_batch:
            return

        output = torch.tensor(reward_batch).to(self.device)
        next_state_tensor = torch.concatenate(new_state_batch, axis=0)
        next_state_q_values = self._network(next_state_tensor)

    def eval(self):
        pass