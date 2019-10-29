# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json

import pandas as pd
import numpy as np

from abc import ABCMeta, abstractmethod
from typing import Union, Callable, List, Dict

import neat

from tensortrade.environments.trading_environment import TradingEnvironment
from tensortrade.features.feature_pipeline import FeaturePipeline
from tensortrade.strategies import TradingStrategy
from IPython.display import clear_output



class NeatTradingStrategy(TradingStrategy):
    """A trading strategy capable of self tuning, training, and evaluating using the NEAT Neuralevolution."""

    # todo: pass in config file
    def __init__(self, environment: TradingEnvironment, neat_config: str, **kwargs):
        """
        Arguments:
            environment: A `TradingEnvironment` instance for the agent to trade within.
            neat_sepc: A specification dictionary for the `Tensorforce` agent's model network.
            kwargs (optional): Optional keyword arguments to adjust the strategy.
        """
        self._environment = environment

        self._max_episode_timesteps = kwargs.get('max_episode_timesteps', None)
        self._neat_config_filename = neat_config
        self._config = self.load_config()


    @property
    def environment(self):
        return self._environment

    def load_config(self):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         self._neat_config_filename)
        # config.genome_config.num_inputs = len(self._environment._exchange.generated_columns)
        # config.genome_config.input_keys = [-i - 1 for i in range(config.genome_config.num_inputs)]
        return config

    def restore_agent(self, path: str, model_path: str = None):
        raise NotImplementedError

    def save_agent(self, path: str, model_path: str = None, append_timestep: bool = False):
        raise NotImplementedError

    def _finished_episode_cb(self) -> bool:
        n_episodes = runner.episode
        n_timesteps = runner.episode_timestep
        avg_reward = np.mean(runner.episode_rewards)

        print("Finished episode {} after {} timesteps.".format(n_episodes, n_timesteps))
        print("Average episode reward: {})".format(avg_reward))

        return True

    def tune(self, steps: int = None, episodes: int = None, callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        raise NotImplementedError

    def _eval_population(self, genomes, config):
        for genome_id, genome in genomes:
            print("*", end = '')
            self.eval_genome(genome)
        print(' ')
        # clear_output()

    def eval_genome(self, genome):
        # Initialize the network for this genome
        net = neat.nn.RecurrentNetwork.create(genome, self._config)

        # calculate the steps and keep track of some intial variables
        steps = len(self._environment._exchange.data_frame)
        steps_completed = 0
        obs, dones = self._environment.reset(), [False]
        performance = {}

        # we need to know how many actions we are able to take
        actions = self._environment.action_strategy.n_actions

        # set inital reward
        genome.fitness = 0.0
        # walk all timesteps to evaluate our genome
        while (steps is not None and (steps == 0 or steps_completed < (steps))):
            # Get the current data observation
            current_dataframe_observation = self._environment._exchange.data_frame[steps_completed:steps_completed+1]

            # transform as needed
            current_dataframe_observation = current_dataframe_observation.values.flatten()

            # activate() the genome and calculate the action output
            output = net.activate(current_dataframe_observation)

            # action at current step
            action = int(output[0] * actions)

            # feed action into environment to get reward for selected action
            obs, rewards, dones, info = self.environment.step(action)

            # feed rewards to NEAT to calculate fitness.
            genome.fitness += rewards
            steps_completed += 1

            exchange_performance = info.get('exchange').performance
            performance = exchange_performance if len(exchange_performance) > 0 else performance

            if dones:
                break

    def profit_report(slef):
        print("Average Trades:", )



    def run(self, generations: int = None, testing: bool = True, episode_callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:

        # create population
        pop = neat.Population(self._config)

        # add reporting
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.Checkpointer(5))

        # Run for up to 300 generations.
        winner = pop.run(self._eval_population, generations)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        # print('\nOutput:')

        # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')

        return [self._environment._exchange.performance, winner, stats]
