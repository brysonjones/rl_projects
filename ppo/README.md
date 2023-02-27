# Proximal Policy Optimization (PPO) Implementation

## Overview
This project focuses on implementing PPO from scratch, by referencing the original paper, 
and various other references. The algorithm is implemented in Python, and is  
benchmarked in various simple environments

## Resources
### Papers
* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
    * This is noted as the original PPO paper, which references the relative complexity of 
      TRPO, and how PPO is able to achieve similar levels of performance with less effort
* [Towards Delivering a Coherent Self-Contained Explanation of Proximal Policy Optimization](https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf)
    * A nice deep dive into the different pieces of PPO in detail, and breaks down different
      details on continuous vs. discrete environments
* [HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION](https://arxiv.org/pdf/1506.02438.pdf)
    * A paper that described GAE, which was the most helpful form advantage calculation I found

### Articles
* [Hugging Face - A2C description](https://huggingface.co/blog/deep-rl-a2c)
    * Not directly related to PPO, but helpful for understanding the bias-variance
    tradeoff, along with the meaning of and calculation of the advantage value
* [Hugging Face - PPO description](https://huggingface.co/blog/deep-rl-ppo)


## The Algorithm
* PPO is an on-polcy algorithm, meaning that it evaluates and improves the same policy that
  that is actively being used to select actions
* The goal of PPO is to achieve similar levels of performance (or higher), than Trust-Region
  Policy Optimization (TRPO), but with simpler implementation and less computational cost
* Effectively, PPO attempts to lower the variance seen in vanilla polict gradient 
  optimization methods (such as REINFORCE), by clipping the objective function, and limiting
  how far away from the current policy a new policy can be

## Implementation
* The actor and critic models are both implemented in PyTorch, and are simple MLPs
* Every iteration, a fixed number of steps are rolled out with the current version of the policy and stored into torch tensors
  * this number of steps is a hyperparameter that can be set
  * if the environment terminates, it is reset and execution continues until the total number of samples equals the set number of steps
* During the training process this set of samples is shuffled, batched, and used to train the actor and critic networks
  * the training data is re-used for a set number of epochs, which is a hyperparameter


## Results
### CartPoleV1 Environment


## Takeaways
* PPO is extremely sensitive to implementation details -- I fought for quite a while getting the first working version of this code together
* The two things that seemed to finally get my version working were
  * layer initialization
  * GAE Advantage estimation

