import subprocess
import numpy 
from stable_baselines3 import PPO
from PPO_Env import LanderEnv
from callback import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.monitor import Monitor

savedir = r"C:\Users\markm\source\repos\Lander\Lander\pythonfiles\PPOMODELS\ModelSaves2"
gamedir = r"C:\Users\markm\source\repos\Lander\x64\Debug\Lander.exe"
loaddir = r"C:\Users\markm\source\repos\Lander\Lander\pythonfiles\PPOMODELS\ModelSaves2"
logdir = r"C:\Users\markm\source\repos\Lander\Lander\pythonfiles\CallbackLog"

global simulation
simulation = 1

#envoronment
env = LanderEnv(simulation_number=simulation)
env = Monitor(env, logdir)

#load model
loadedModel = PPO.load(loaddir, env=env)
callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=logdir, verbose = 1)

# run the mars lander program
def runLander():
    """runs the lander program"""
    file = subprocess.run([gamedir])
    

def trainPPO(model, timesteps, saveDIR):
    """takes a model to train, if None, creates a model, trains for a set number of timesteps"""
    if model == None:
        model = PPO("MlpPolicy", env=env, verbose=2, tensorboard_log="./board/")
    model.learn(total_timesteps = timesteps, callback=callback)
    model.save(saveDIR)

#train model
trainPPO(None, 100000, savedir)
