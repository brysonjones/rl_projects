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
### Articles
* [Hugging Face - A2C description](https://huggingface.co/blog/deep-rl-a2c)
    * Not directly related to PPO, but helpful for understanding the bias-variance
    tradeoff, along with the meaning of and calculation of the advantage value
* [Hugging Face - PPO description](https://huggingface.co/blog/deep-rl-ppo


## The Algorithm
* PPO is an on-polcy algorithm, meaning that it evaluates and improves the same policy that
  that is actively being used to select actions
* The goal of PPO is to achieve similar levels of performance (or higher), than Trust-Region
  Policy Optimization (TRPO), but with simpler implementation and less computational cost
* Effectively, PPO attempts to lower the variance seen in vanilla polict gradient 
  optimization methods (such as REINFORCE), by clipping the objective function, and limiting
  how far away from the current policy a new policy can be

## Implementation
### Hyperparameters

### Models

### Data Tracking

### Training

#### Terminal States




## Results
### CartPoleV1 Environment


## Takeaways