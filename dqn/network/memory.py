
from collections import deque
import random

# a class to represent the Experience Replay Buffer
class Memory():
    def __init__(self, max_length=1000):
        self.transtion_list = deque(maxlen=max_length)

    # append a tuple to the replay buffer
    def add_transition(self, new_state, prev_state, action, reward, done):
        self.new_state_list.append(new_state)
        self.prev_state_list.append(prev_state)
        self.action_list.append(action)
        self.reward_list.append(reward)
        self.done_state_list.append(done)

    # return uniformly random sampled batch of transitions stored in memory replay buffer
    def get_samples(self, sample_size):
        new_state_batch = []
        prev_state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        for i in range(sample_size):
            sample_idx = random.randint(0, len(self.transtion_list))
            new_state_batch.append(self.new_state_list[sample_idx])
            prev_state_batch.append(self.prev_state_list[sample_idx])
            action_batch.append(self.action_list[sample_idx])
            reward_batch.append(self.reward_list[sample_idx])
            done_batch.append(self.done_state_list[sample_idx])

        return prev_state_batch, action_batch, reward_batch, new_state_batch, done_batch
