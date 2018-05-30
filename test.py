# -*- coding: utf-8 -*-
"""
Created on Mon May 28 22:11:15 2018

@author: sarac
"""

from EURUSDagent import DQNAgent
import datetime
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# init conts.
############################################

EPISODES = 1
MARGIN = 1000
state_size = 60

###########################################

#start_index = 45    #2010.01.01 00:00
#end_index = 3161+1  #2011.12.30 20:00

start_index = 3162 - state_size    # 2011.12.16 16:00 start at 2012.01.02 00:00
end_index = 4662+1  #2012.12.30 16:00
dataset = pd.read_csv('EURUSD_4H.csv')
train_data = dataset.iloc[start_index:end_index,5:6]

train_data = np.array(train_data)
state_size = 60
X_train = [] 
all_index = end_index-start_index
for i in range(state_size, all_index):
    X_train.append(train_data[i-state_size:i,0])
X_train = np.array(X_train)


class TrainEnvironment:
    def __init__(self, data, num_index):
        self.train_data = data
        self.train_index = 0 
        self.end_index = num_index-1
        self.loss_limit = 0.3 # force sell 
        
        self.profit = 0
        self.reward = 0
        self.mem_reward = 0
        
        # portfolio 
        self.cost_price = 0 
        self.mem_action = 0
        
    def reset(self):
        self.train_index = 0 
        self.profit = 0
        self.reward = 0 
        self.cost_price = 0 
        self.mem_action = 0
        self.mem_reward = 0
        
        return [self.train_data[self.train_index]]
    
    def get_action(self,action):
        if action == 0 :
            # buy 
            return 1
        elif action == 1 : 
            # sell 
            return -1
        else : 
            # noaction 
            return 0 
    
    def calculate_reward(self, action):
        action = self.get_action(action)
        current_price = self.train_data[self.train_index,59:60]
        if action == self.mem_action :
            self.profit = action*(current_price - self.cost_price)
            self.reward = self.mem_reward + self.profit
        else :  
            if action == 0 : 
                self.profit = self.mem_action*(current_price - self.cost_price)    
            else :
                self.profit = current_price*(-0.001) + self.mem_action*(current_price - self.cost_price)
            self.reward += self.profit
            self.mem_reward = self.reward 
            self.cost_price = current_price
            self.mem_action = action

    def done_check(self):
        if self.cost_price != 0 : 
            loss = -self.loss_limit*self.cost_price
        else : 
            loss = -self.loss_limit*self.train_data[self.train_index,59:60]
        if self.train_index + 1 == self.end_index :
            if self.reward > 0 : 
                if self.reward <= 0.001 :
                    self.reward = -1
            print('Full End !')
            return True 
        elif self.reward <= loss : 
            print('------------------------------------------------------------')
            print('loss limit: ', loss)
            print('reward : ', self.reward)
            print('Cut Loss !')
            self.reward = -3
            return True
        else :
            return False
        
    def step(self,action):
        skip = 1
        self.train_index += skip
        if self.train_index >= self.end_index-1 : 
            self.train_index = self.end_index-1 
        ns = [self.train_data[self.train_index]]
        self.calculate_reward(action)
        done = self.done_check()
        return ns, self.reward*MARGIN, done

#########################################################################################################
# Train     
#########################################################################################################         
def watch_result(episode ,s_time, e_time, c_index, all_index, action, reward, profit):
    print('-------------------- Check -------------------------')
    print('start time: ' + s_time)  
    print('counter : ', c_index,'/', all_index,' of episode : ', episode, '/', EPISODES)
    print('action : ', action)
    print('current profit : ', profit*MARGIN)
    print('reward (all profit): ', reward)
    print('end_time: ' + e_time)
    print('-------------------End Check -----------------------')

    
if __name__ == "__main__":
    
    agent = DQNAgent(state_size)
    agent.load("agent_model.h5")
    num_index = all_index - state_size
    env = TrainEnvironment(X_train, num_index)
    batch_size = 32 
    test_profit = []
    
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, (1, state_size, 1))
        test_profit = [] 
        for t in range(end_index-start_index):
            start_time = str(datetime.datetime.now().time())
            action = agent.act(state, False)    # test 
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, (1,state_size,1))
            agent.remember(state, action, reward, next_state, done)
            state = next_state       
            if done:
                agent.update_target_model()
                print('----------------------------- Episode Result -----------------------')
                print("episode: {}/{}, time: {}, e: {:.2}"
                      .format(e, EPISODES, t, agent.epsilon))
                print('----------------------------- End Episode --------------------------')
                break
            """
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            """
            end_time = str(datetime.datetime.now().time())
            
            watch_result(e , start_time, end_time, env.train_index, end_index-start_index, env.get_action(action), reward ,env.profit)  
            
            test_profit.append(env.reward*1000)
    test_profit = np.array(test_profit)
    
    # agent.save("agent_model.h5")
    plt.plot(test_profit, color = 'blue', label = 'Profit')
    plt.title('Profit')
    plt.xlabel('Time')
    plt.ylabel('Profit')
    plt.legend()
    plt.show()

                      
    