
import matplotlib.pyplot as plt

def plot_rolling_rewards(rewards):
    plt.plot(rewards)
    plt.ylabel('Episode #')
    plt.ylabel('Rolling Reward')
    plt.show()