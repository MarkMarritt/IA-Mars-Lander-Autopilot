# IA-Mars-Lander-Autopilot
Implementation of PPO using the stable baselines 3 library

Primarily a wrapper of the gymnasium environemt for the C++ mars lander environment supplied for coursework. 
The PPO model however takes far too long to train in the end so was abandoned half way through. 
Would be far cleaner if sockets or apis were used to communicate between C++ and python code, or just all written in C++ but machine learning support is much better on python. 

Note still has a crashing issue that needs to be fixed. 
