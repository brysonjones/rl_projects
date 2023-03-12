# Deep Deterministic Policy Gradient (DDPG)

## Overview
This project focuses on implementing DDPG from scratch, by referencing the original paper, 
and various other references. The algorithm is implemented in Python, and is  
benchmarked in various simple environments against existing implementations

## Resources
### Papers


### Articles

### Libraries

## The Algorithm
* DDPG is an off-policy algorithm, meaning it learns from data gathered from sources other than it's current policy
* It can only be utilized in environments with continuous action spaces
* Two big implementation details that are important for DDPG:
  * The Replay Buffers -- this is where previous experiences are stored to be trained on later, and the challenges is balancing keeping enough data not to overfit to one section of experiences, but not running out of memory/storage
  * Target Networks -- When trying to minimize the Mean-Squared Bellman Error (MBSE) loss, we are trying to make the Q function more like this target, but the target depends on the same set of paramters that we are updating, which leads to stability issues during training
    * To mitigate this, a second network is created, where this networks parameters are either frozen in time or updated on a slower bases
    * In common literature, this target network ahs it's parameters updated using `polyak` averaging 


## Implementation


## Results


## Takeaways
