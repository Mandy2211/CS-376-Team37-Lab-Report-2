import numpy as np
import matplotlib.pyplot as plt

class nonstatbandit:
    def __init__(self,meanreward=0):
        self.meanreward=meanreward
        self.random_walk_std=0.01
        
    def pull(self):
        reward=np.random.normal(self.meanreward, 1.0)
        self.meanreward += np.random.normal(0,self.random_walk_std)
        return reward

class epgreedy:
    def __init__(self, n_arms, epsilon,alpha=0.1):
        self.n_arms=n_arms
        self.epsilon=epsilon
        self.counts=np.zeros(n_arms)
        self.values=np.zeros(n_arms)
        self.alpha=alpha
        self.action_history=[]
        self.reward_history=[]
    
    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.values)
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.values[chosen_arm] += self.alpha * (reward - self.values[chosen_arm])
        self.action_history.append(chosen_arm)
        self.reward_history.append(reward)



def simulation(n_steps,bandits,epsilon):
    n_arms=len(bandits)
    agent=epgreedy(n_arms,epsilon)
    
    cumulative_rewards=np.zeros(n_steps)
    meanrewards_history=np.zeros((n_steps, n_arms))
    
    for t in range(n_steps):
       
        for i, bandit in enumerate(bandits):
            meanrewards_history[t, i]=bandit.meanreward    
        chosen_arm=agent.select_arm()
        reward=bandits[chosen_arm].pull()
        agent.update(chosen_arm, reward)
        cumulative_rewards[t]=reward
            
    return agent, cumulative_rewards, meanrewards_history


n_steps=10000
epsilon=0.1
n_arms=10
non_stationary_bandits=[nonstatbandit() for _ in range(n_arms)]
agent_ns, rewards_ns,meanrewards_history=simulation(n_steps,non_stationary_bandits,epsilon)

plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
window=50  
moving_avg=np.convolve(rewards_ns, np.ones(window)/window, mode='valid')
plt.plot(moving_avg)
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Average Reward Over Time (Moving Average)')
plt.show()


print("\nNon-stationary Bandit Results:")
print(f"Total Reward: {np.sum(rewards_ns)}")
print(f"Average Reward: {np.mean(rewards_ns):.3f}")
print("\nFinal Arm Statistics:")
for i in range(n_arms):
    print(f"Arm {i+1}:")
    print(f"  Final Mean Reward: {meanrewards_history[-1, i]:.3f}")
    print(f"  Estimated Value: {agent_ns.values[i]:.3f}")
    print(f"  Times Selected: {int(agent_ns.counts[i])}")