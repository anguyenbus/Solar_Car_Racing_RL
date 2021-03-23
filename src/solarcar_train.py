import json
import os

import gym
import ray
from ray.tune import run_experiments


from ray.tune.registry import register_env
from solarcar_env import SolarCarEnv


from sagemaker_rl.ray_launcher import SageMakerRayLauncher
        


class MyLauncher(SageMakerRayLauncher):
        
          
    def register_env_creator(self):
        env_name = "SolarCarEnv-v0"
        register_env(env_name, lambda env_config: SolarCarEnv())


    def get_experiment_config(self):
        return {
          "training": {
            "env": "SolarCarEnv-v0",
            "run": "PPO",
            "stop": {
              "training_iteration": 5,
            },
            "config": {
              "gamma": 0.995,
              "kl_coeff": 1.0,
              "num_sgd_iter": 20,
              "lr": 0.0005,
              "sgd_minibatch_size": 1500,
              "train_batch_size": 25000,
              "monitor": True,  # Record videos.
              "model": {
                "free_log_std": True
              },
              "num_workers": (self.num_cpus-1),
              "num_gpus": self.num_gpus,
              "batch_mode": "truncate_episodes"
            }
          }
        }

if __name__ == "__main__":


    MyLauncher().train_main()