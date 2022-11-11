
import numpy as np
import cv2

from network.memory import Memory

class MainModel():
    def __init__(self, network, optimizer):
        self._network = network
        self._optimizer = optimizer
        self._replay_memory = Memory()

    def get_action(self, current_obs):
        pass

    def update_state(self, prev_state, action, reward, new_obs):
        '''
        prev_state: previous state that had already been preprocessed
        action: action take for the transition
        reward: reward received during the transition
        new_obs: observation after transition that has not yet been preprocessed
        '''
        processed_frame = self.preprocess(new_obs)

        # initialization step at the beginning of an episode
        if prev_state == None:
            # set the prev_state to be a repeating sequence of the new observation
            # TODO: should this be modular in length?
            new_processed_state = np.dstack((processed_frame, processed_frame, processed_frame, processed_frame))
            return new_processed_state  # don't need to add to replay memory at the beginning 
        else:
            # add and shift state frame with newly preprocessed observation data
            new_processed_state = np.dstack((prev_state, processed_frame))[1::]

        self._replay_memory.add_transition(prev_state, action, reward, new_processed_state)
        return new_processed_state
    

    def preprocess(self, current_obs):
        # convert image to gray
        image_gray = cv2.cvtColor(current_obs, cv2.COLOR_RGB2GRAY)

        # determine square side length
        height, width = image_gray.shape
        slice = int((width-height)/2)
        image_square = image_gray[:, slice:-slice]

        # resize
        image_scaled = cv2.resize(image_square, (84, 84), interpolation=cv2.INTER_NEAREST)

        return image_scaled

    def learn(self):
        pass

    def eval(self):
        pass