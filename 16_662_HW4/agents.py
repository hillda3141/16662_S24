# Author: Vibhakar Mohta (vmohta@cs.cmu.edu)

import numpy as np
import matplotlib.pyplot as plt

class RandomAgent():        
    def __init__(self):
        self.available_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
    # Choose a random action
    def get_action(self, state, explore=True):
        return self.available_actions[np.random.randint(len(self.available_actions))]
        

class QAgent():
    def __init__(self, environment, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.environment = environment
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # TODO: Initialize the Q-table with random values
        self.q_table = dict()
        for i in range(self.environment.height):
            for j in range(self.environment.width):
                self.q_table[(i,j)] = dict()
                if (i,j) != self.environment.gold_location:
                    self.q_table[(i,j)]['UP'] = np.random.uniform(-100,100)
                    self.q_table[(i,j)]['DOWN'] = np.random.uniform(-100,100)
                    self.q_table[(i,j)]['LEFT'] = np.random.uniform(-100,100)
                    self.q_table[(i,j)]['RIGHT'] = np.random.uniform(-100,100)
                else:
                    self.q_table[(i,j)]['UP'] = 0
                    self.q_table[(i,j)]['DOWN'] = 0
                    self.q_table[(i,j)]['LEFT'] = 0
                    self.q_table[(i,j)]['RIGHT'] = 0
        
    def get_action(self, state, explore=True):
        """
        Returns the optimal action from Q-Value table. 
        Performs epsilon greedy exploration if explore is True (during training)
        If multiple optimal actions, chooses random choice.
        """
        # TODO: Implement this function
        if np.random.uniform(0,1) < self.epsilon and explore:
            action_int = np.random.randint(4)
            if action_int == 0:
                action = 'UP'
            elif action_int == 1:
                action = 'DOWN'
            elif action_int == 1:
                action = 'LEFT'
            else:
                action = 'RIGHT'
        else:
            action = 'UP'
            if self.q_table[state]['DOWN'] > self.q_table[state][action]:
                min_action = 'DOWN'
            if self.q_table[state]['LEFT'] > self.q_table[state][action]:
                min_action = 'LEFT'
            if self.q_table[state]['RIGHT'] > self.q_table[state][action]:
                min_action = 'RIGHT'

        return action
    
    def update(self, state, reward, next_state, action):
        """
        Updates the Q-value tables using Q-learning
        """
        # TODO: Implement this function
        self.q_table[state][action] = self.q_table[state][action] + self.alpha * (reward + self.gamma * self.q_table[next_state][self.get_action(state,explore=False)] - self.q_table[state][action])
    
    def visualize_q_values(self, SCALE=100):
        """
        In the grid, plot the arrow showing the best action to take in each state.
        """
        plt.figure()
        # Display grid (red as bomb, yellow as gold)
        img = np.zeros((self.environment.height*SCALE, self.environment.width*SCALE, 3)) + 0.01
        for bomb_location in self.environment.bomb_locations:
            img[bomb_location[0]*SCALE:bomb_location[0]*SCALE+SCALE, bomb_location[1]*SCALE:bomb_location[1]*SCALE+SCALE] = [1, 0, 0]
        img[self.environment.gold_location[0]*SCALE:self.environment.gold_location[0]*SCALE+SCALE, self.environment.gold_location[1]*SCALE:self.environment.gold_location[1]*SCALE+SCALE] = [1, 1, 0]
        
        # Display best actions (print as text), show as blue arrow
        for i in range(self.environment.height):
            for j in range(self.environment.width):
                # skip terminal states
                if (i, j) in self.environment.terminal_states:
                    continue
                state = (i, j)
                best_action = self.get_action(state)
                if best_action == 'UP':
                    plt.arrow(j*SCALE+SCALE//2, i*SCALE+SCALE//2, 0, -40, color='blue', head_width=10)
                elif best_action == 'DOWN':
                    plt.arrow(j*SCALE+SCALE//2, i*SCALE+SCALE//2, 0, 40, color='blue', head_width=10)
                elif best_action == 'LEFT':
                    plt.arrow(j*SCALE+SCALE//2, i*SCALE+SCALE//2, -40, 0, color='blue', head_width=10)
                elif best_action == 'RIGHT':
                    plt.arrow(j*SCALE+SCALE//2, i*SCALE+SCALE//2, 40, 0, color='blue', head_width=10)
        
        plt.imshow(img)
        plt.xticks(np.arange(0, self.environment.width*SCALE, SCALE), np.arange(0, self.environment.width, 1))
        plt.yticks(np.arange(0, self.environment.height*SCALE, SCALE), np.arange(0, self.environment.height, 1))
        plt.title('Best Q-Value Actions')
        plt.savefig('q_values.png')