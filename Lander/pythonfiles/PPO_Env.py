import gymnasium
from gymnasium import spaces
import numpy as np
import time
from stable_baselines3.common.env_checker import check_env
from datetime import datetime


# box parameters, corresponds to the output data from the cpp file
boxLowIn = np.array([-10000000, -10000000, -10000000, -10000, -10000, -10000, 0, 0, 0, 0, 0, 0, 0, 0])
boxHighIn = np.array([10000000, 10000000, 10000000, 10000, 10000, 10000, 360, 360, 360, 1, 1, 1, 1, 1,])
dir = r"C:\Users\markm\source\repos\Lander\Lander\landerOutputs.txt"
outDir = r"C:\Users\markm\source\repos\Lander\Lander\pythonOutputs.txt"
FUEL = -4
THROTTLE = 9
CRASHED = -1
LANDED = -2
MARS_RADIUS = 3386000

class LanderEnv(gymnasium.Env):
    """environment wrapper of gym.Env so it can be trained easily using stablebaslines ppo algorithm"""
    metadata = {"render_modes": [None], "render_fps": 10}
    

    def __init__(self, simulation_number):
        self.observation_space = spaces.Box(low= boxLowIn, high = boxHighIn)  # the 14 numbers from the .txt file to be loaded
        self.action_space = spaces.Discrete(5)  # nothing, +/- 0.1 thrust, parachute, stabiliser
        self.simulation_number = int(simulation_number)
        self.outputState = [0,0,0] # thrust, parachute, stabiliser
        self.lastAlt = 10000000
        self.lastDistance = 10000000 + MARS_RADIUS

    def getObs(self):
        """retrieves observation data from text file from the cpp file
        output is a list"""
        self.lastAlt = self.lastDistance - MARS_RADIUS
        with open(dir, "r") as f: # reads last line in a file
            line = [float(f) for f in f.readlines()[-1].split()]
        return np.array(line, dtype="float32")
    
    def output(self, State):
        """outputs input character to text file"""
        if isinstance(State, int):
            outState = "switch " + str(State)
        elif None in State:
            pass
        else:
            outState = " ".join([str(x) for x in State])
        if " " in outState:
            with open(outDir, "a") as f:
                f.write(outState + "\n")
        
    
    def reset(self, seed=None):
        """resets the simulation, and returns an obs and distance from the centre of planet"""
        self.output(self.simulation_number)
        obs = self.getObs()
        distance = {"distance" : None}
        self.outputState = [0,0,0]
        open(dir, "w").close()
        time.sleep(0.02)
        return obs, distance
    

    def calcReward(self, obs):
        """calculates reward"""
        reward = 0
        self.lastDistance = np.linalg.norm(obs[:3])
        alt = self.lastDistance - MARS_RADIUS
        vel = np.linalg.norm(obs[3:6])
        reward -= 0.5 * vel if alt > self.lastAlt else 0
        reward -= 0.5 * ((vel - 0.7 - 0.01*alt)**2)**(1/3) # should hopefully punishish deviation from speed 
        reward += 1 - 2*(0.4 - obs[THROTTLE])**2
        reward -= vel * 0.2 if obs[CRASHED] == 1 else 0
        reward += 5 + (10 * obs[FUEL]) if obs[LANDED] == 1 else 0
        return reward
        
    def step(self, action):
        """takes in an action, should be of the form 0 to 4
        returns obs, reward, terminated, ignore, info"""
        time.sleep(0.001)
        if action == 0:
            pass
        elif action == 1:
            self.outputState[0] += 0.1
            if self.outputState[0] > 1:
                self.outputState[0] = 1
        elif action == 2:
            self.outputState[0] -= 0.1
            if self.outputState[0] < 0:
                self.outputState[0] = 0
        elif action == 3:
            self.outputState[1] = 1 - self.outputState[1] # parachute
        elif action == 4:
            self.outputState[2] = 1 - self.outputState[2] # stabiliser
        self.output(self.outputState)
        obs = self.getObs()
        reward = self.calcReward(obs=obs)
        terminated = True if 1 in obs[-2:] else False
        print(reward)
        return obs, reward, terminated, False, {"distance" : None}
