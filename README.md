# Agent for EURUSD 

state : 60 candlesticks (close price of 4 hours EURUSD) 

reward : all profit 

action(output) : This DQN has 3 actions 1 = buy , 0 = noact , -1 = sell 

To train this agent, run "train.py".

During training, it will print current reward, action of every train-step.

Afterthat it will save weights at "agent_model.h5"


To test this agent, run "test.py".

It will load trained-weights from "agent_model.h5".

During testing, it will print current reward, action of every test-step.

