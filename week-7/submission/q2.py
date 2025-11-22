import numpy as np
import matplotlib.pyplot as plt

class bibandit:
    def __init__(self, name, probabilities):
        self.name=name
        self.probabilities=probabilities
    
    def pull(self, action):
        if action < 0 or action >= len(self.probabilities):
            raise ValueError("Invalid action")
        return np.random.random() < self.probabilities[action]

class epgreedy:
    def __init__(self, n_arms, epsilon):
        self.n_arms=n_arms
        self.epsilon=epsilon
        self.counts=np.zeros(n_arms)
        self.values=np.zeros(n_arms)
        self.action_history=[]
        self.reward_history=[]
    
    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.values)
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n=self.counts[chosen_arm]
        value=self.values[chosen_arm]
        self.values[chosen_arm]=((n - 1) / n) * value + (1 / n) * reward
        self.action_history.append(chosen_arm)
        self.reward_history.append(reward)

def simulation(environment, n_steps, epsilon):
    agent=epgreedy(len(environment.probabilities), epsilon)
    
    cumulative_rewards=np.zeros(n_steps)
    optimal_actions=0
    optimal_arm=np.argmax(environment.probabilities)
    
    for t in range(n_steps):
        chosen_arm=agent.select_arm()
        reward=environment.pull(chosen_arm)
        agent.update(chosen_arm, reward)
        
        cumulative_rewards[t]=reward
        if chosen_arm == optimal_arm:
            optimal_actions += 1
            
    return agent, cumulative_rewards, optimal_actions

def plot_results(environment, agent, rewards, n_steps):
    plt.figure(figsize=(15, 10))
   
    plt.subplot(2, 2, 1)
    actions=np.array(agent.action_history)
    for arm in range(len(environment.probabilities)):
        plt.plot(np.cumsum(actions == arm) / (np.arange(len(actions)) + 1),
                 label=f'Arm {arm+1} (p={environment.probabilities[arm]})')
    plt.xlabel('Steps')
    plt.ylabel('Selection Probability')
    plt.title(f'{environment.name}: Arm Selection Probability Over Time')
    plt.legend()


    plt.subplot(2, 2, 2)
    cumulative_average=np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
    plt.plot(cumulative_average)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title(f'{environment.name}: Average Reward Over Time')


    optimal_actions=sum(np.array(agent.action_history) == np.argmax(environment.probabilities))
    print(f"\n{environment.name} Results:")
    print(f"Total Reward: {np.sum(rewards)}")
    print(f"Average Reward: {np.mean(rewards):.3f}")
    print(f"Optimal Action Percentage: {100 * optimal_actions / n_steps:.1f}%")
    print("\nArm Statistics:")
    for i in range(len(environment.probabilities)):
        print(f"Arm {i+1}:")
        print(f"  True Probability: {environment.probabilities[i]:.2f}")
        print(f"  Estimated Value: {agent.values[i]:.3f}")
        print(f"  Times Selected: {int(agent.counts[i])}")

def main():
 
    n_steps=1000
    epsilon=0.1
    bandit_A=bibandit("Bandit A",[0.1, 0.2])  
    bandit_B=bibandit("Bandit B",[0.8, 0.9]) 
    for environment in [bandit_A, bandit_B]:
        agent, rewards,_=simulation(environment, n_steps, epsilon)
        plot_results(environment, agent, rewards, n_steps)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()