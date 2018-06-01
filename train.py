# -*- coding: utf-8 -*-
"""
Created on Mon May 28 22:11:15 2018

@author: sarac
"""

from EURUSDagent import DQNAgent
import datetime
import numpy as np
import pandas as pd 

EPISODES = 300
MARGIN = 1000

start_index = 4664    #2013.01.02 12:00
end_index = 8322+1  #2016.06.01 00:00
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
            self.reward = self.profit + self.mem_reward
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
                if self.reward <= 0.05*self.train_data[self.train_index,59:60]:
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
        skip = 6  # half day 
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
    #agent.load("agent_model.h5")
    
    num_index = all_index - state_size
    env = TrainEnvironment(X_train, num_index)
    
    batch_size = 15 # Train Every 3 weeks data 
    best_reward = -300
     
    same_act_limit = 10 
    
    for e in range(EPISODES):
        last_action = -100
        same_action = 0
        
        state = env.reset()
        state = np.reshape(state, (1, state_size, 1))  
        for t in range(end_index-start_index):
            start_time = str(datetime.datetime.now().time())
            
            if same_action >= same_act_limit : 
                print('random actions')
                action = agent.act(state, random_action = True)
                same_action -= 1
            else :
                action = agent.act(state)
            
            if action == last_action :
                same_action += 1 
            else :
                same_action -= 1 
                last_action = action
                if same_action < -10 :
                    same_action = -10 
             
            next_state, reward, done = env.step(action)
            
            next_state = np.reshape(next_state, (1,state_size,1))
            agent.remember(state, action, reward, next_state, done)
            state = next_state 
            if done:
                agent.update_target_model()
                print('----------------------------- Episode Result -----------------------')
                print("episode: {}/{}, time: {}, e: {:.4}"
                      .format(e+1, EPISODES, t, agent.epsilon))
                print('----------------------------- End Episode --------------------------')
                if reward >= best_reward :
                    best_reward = reward
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            end_time = str(datetime.datetime.now().time())
            watch_result(e+1 , start_time, end_time, env.train_index, end_index-start_index, env.get_action(action), reward ,env.profit) 
            print('same action counter : ', same_action, '/', same_act_limit) 
        agent.save("agent_model.h5")
        
    #agent.save("agent_model.h5")

    print('train done')
    print('BEST RESULT ==================================')
    print("best reward : ", best_reward)



