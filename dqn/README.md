# Deep Q-Network (DQN) Implementation

## Overview
This project was focused on implementing a DQN reinforcement learning method, utilizing purely visual input to train and evaluate the model. The methodology is tested on various environments available in OpenAI's `gym` API.

## Resources
### Papers
* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)
  * The original DQN paper from DeepMind that revitalized Q-Learning in 2013
* [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)
* [Double Q-Learning](https://papers.nips.cc/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf)

### Articles

## Concepts
### Preprocessing
#### Part #1
The first part of pre-precessing is to trim down the information size of each observation such that the image is square and converted to gray scale


An example of this process can be seen below:

**Raw Frame:**

**Square Frame:** 

**Grey-Scaled Frame:**

#### Part #2
The second part is to only utilize the last four frames of observation history, to mitigate the need to handle varying lengths of inputs.

We utilize the last four frames to provide context to the Q-network about how the system is changing over time. A single frame would only show where objects are in space at that time-step, but not what vector they are traveling along, their speed, if they are stationary, etc.

### Memory Replay Buffer

## Implementation
### Q-Network Architecture
### Software Archicecture
### Results

## Discussion
### Successes
### Shortcomings
### Future Work
