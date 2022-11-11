
from collections import deque
import random

# a class to represent the Experience Replay Buffer
class Memory():
    def __init__(self, max_length=1000):
        self.transtion_list = deque(maxlen=max_length)

    # append a tuple to the replay buffer
    def add_transition(self, prev_state, action, reward, new_state):
        self. transtion_list.append((prev_state, action, reward, new_state))

    # return uniformly random sampled batch of transitions stored in memory replay buffer
    def get_samples(self, sample_size):
        batch = []
        for i in range(sample_size):
            sample = self.transtion_list[random.randint(0, len(self.transtion_list))]
            batch.append(sample)

        return batch
