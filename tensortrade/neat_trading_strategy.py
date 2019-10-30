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

from statistics import mode, stdev, StatisticsError

from abc import ABCMeta, abstractmethod
from typing import Union, Callable, List, Dict

import neat
from collections import Counter

from tensortrade.environments.trading_environment import TradingEnvironment
from tensortrade.features.feature_pipeline import FeaturePipeline
from tensortrade.strategies import TradingStrategy
from termcolor import colored as c
from IPython.display import clear_output
import math
import random

import matplotlib.pyplot as plt

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
        self._genome_performance = {}
        self._learn_to_trade_theshold = kwargs.get('learn_to_trade_theshold', 300)

    @property
    def environment(self):
        return self._environment

    def load_config(self):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         self._neat_config_filename)
        config.genome_config.num_inputs = len(self._environment.exchange.data_frame.columns)
        config.genome_config.input_keys = [-i - 1 for i in range(config.genome_config.num_inputs)]
        return config

    def restore_agent(self, path: str, model_path: str = None):
        raise NotImplementedError

    def save_agent(self, path: str, model_path: str = None, append_timestep: bool = False):
        raise NotImplementedError

    def _finished_episode_cb(self) -> bool:
        n_episodes = runner.episode
        n_timesteps = runner.episode_timestep
        avg_reward = np.mean(runner.episode_rewards)
        print("Average Trades:", self.exchange.performance[-10:] )
        print("Trades: ", mean(self._genome_performance["trades"]))

        print("Finished episode {} after {} timesteps.".format(n_episodes, n_timesteps))
        print("Average episode reward: {})".format(avg_reward))

        return True

    def tune(self, steps: int = None, episodes: int = None, callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        raise NotImplementedError

    def _eval_population(self, genomes, config):
        # find a window to evaluate all genomes on
        data_frame_window = 500
        data_frame_length = self.environment.exchange.data_frame.shape[0]
        data_frame_start_tick = random.randint(0, data_frame_length - data_frame_window)
        print("Starting at DF[{}]".format(data_frame_start_tick))
        # show the current plot for the price window.
        # plt.plot(self.environment.exchange.data_frame[data_frame_start_tick:data_frame_start_tick+data_frame_window]['close'])
        # plt.show()

        for genome_id, genome in genomes:
            self._environment.reset()
            # set the current_step to the start of our window
            self.environment._exchange._current_step = data_frame_start_tick
            self.environment._current_step = data_frame_start_tick

            self.eval_genome(genome, data_frame_window)

            p = self._genome_performance[genome.key]
            print("Genome Performance: ", genome.key)

            if p['rewards'] > 0:
                print("Rewards:", c(p['rewards'], 'green'))
            else:
                print("Rewards:", p['rewards'])

            print('Balance:', p['balance'])
            if p['net_worth'] > 10000:
                print("Net Worth:", c(p['net_worth'], 'green'))
            else:
                print("Net Worth:", p['net_worth'])

            print('Steps', p['steps_completed'])
            try:
                print('Most common action', Counter(p['actions']))
            except StatisticsError:
                print('No Action Mode:', p['actions'])
            print('Number of trades:', Counter(self._environment.exchange.trades['type']))
        print(' ')
        # plt.clf()
        clear_output()

    def eval_genome(self, genome, data_frame_window):
        print('---------------------------')

        # Initialize the network for this genome
        net = neat.nn.RecurrentNetwork.create(genome, self._config)
        # calculate the steps and keep track of some intial variables
        steps = len(self._environment._exchange.data_frame)
        steps_completed = 0
        done = False
        actions = self._environment.action_strategy.n_actions

        performance = {"rewards":0, "balance":0, "net_worth":0, "actions": [], "steps_completed":0, 'trades':0}
        self._genome_performance[genome.key] = performance
        # we need to know how many actions we are able to take

        starting_balance = self._environment.exchange.balance

        # set inital reward
        genome.fitness = 0.0

        # walk all timesteps to evaluate our genome
        # while (steps is not None and (steps == 0 or steps_completed < (steps))):
        while(steps_completed < data_frame_window):
            # Get the current data observation
            current_dataframe_observation = self._environment._exchange.data_frame[steps_completed:steps_completed+1]
            current_dataframe_observation = current_dataframe_observation.values.flatten()

            # activate() the genome and calculate the action output
            output = net.activate(current_dataframe_observation)

            # action at current step
            action =  int(self._environment.action_strategy.n_actions/2 * (1 + math.tanh(output[0])))

            # feed action into environment to get reward for selected action
            obs, rewards, done, info = self.environment.step(action)

            # feed rewards to NEAT to calculate fitness.
            genome.fitness += rewards

            # count this as a completed step
            steps_completed += 1

            # stop iterating if we haven't learned to trade or we pass a fitness threshold
            if genome.fitness < -10000:
                print("Learn to trade asshole!")
                done= True



            # if steps_completed > self._learn_to_trade_theshold and len(self._environment.exchange.trades) is 0:
            #     genome.fitness = self._genome_performance[genome.key]['rewards'] = -100000 #lern to trade asshole...
            #
            # # stop iterating if we haven't learned to SELL in the first N timesteps
            # if steps_completed > self._learn_to_trade_theshold and len(self._environment.exchange.trades) is 0:
            #     genome.fitness = self._genome_performance[genome.key]['rewards'] = -100000 #lern to trade asshole...
            #     print("Learn to trade asshole!")
            #     done= True
            #
            # if (
            #     steps_completed > self._learn_to_trade_theshold and
            #     len(self._environment.exchange.trades) is not 0 and
            #     self._environment.exchange.trades.any()
            #     ) :
            #
            #     genome.fitness = self._genome_performance[genome.key]['rewards'] = -100 #lern to trade asshole...
            #     dones= True

            self._genome_performance[genome.key]['rewards'] += rewards
            self._genome_performance[genome.key]['actions'].append(action)
            self._genome_performance[genome.key]['steps_completed'] = steps_completed
            self._genome_performance[genome.key]['trades'] = len(self._environment.exchange.trades)
            self._genome_performance[genome.key]['balance'] = self._environment.exchange.balance
            self._genome_performance[genome.key]['net_worth'] = self._environment.exchange.net_worth

            if done:
                print('-------WE DONE!---------')
                break


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
