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
from copy import deepcopy

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
        self._actions = self._environment.action_strategy.n_actions

        # population controls
        self._pop_size = kwargs.get('pop_size', 20)
        self._max_stagnation = kwargs.get('max_stagnation', 2)
        self._species_elitism = kwargs.get('species_elitism', 1)
        self._elitism = kwargs.get('elitism', 2)

        # network controls
        self._feed_foward = kwargs.get('feed_forward', False)
        self._initial_connection = kwargs.get('initial_connection', 'full_direct')

        # connection controls
        self._enabled_default = kwargs.get('enabled_default', False)
        self._enabled_mutate_rate = kwargs.get('enabled_mutate_rate', 0.01)
        self._conn_add_prob = kwargs.get('conn_add_prob', 0.5)
        self._conn_delete_prob = kwargs.get('conn_delete_prob', 0.1)



        self._neat_config_filename = neat_config
        self._config = self.load_config()

        # catch for custom metrics, this will be moved to a custom stats class eventually
        self._genome_performance = {}
        self._performance_stub = {"rewards":0, "balance":0, "net_worth":0, "actions": [], "steps_completed":0, 'trades':0}

        # If we don't learn to trade, our score will drop due to missed oportunities, if it
        # drops below this level, we should stop iterating.
        self._learn_to_trade_theshold = kwargs.get('learn_to_trade_theshold', -300)

        # how many data points to get from the total set. This allows us to show random data slices
        # to the population in an attempt to avoid overfitting.
        # This is the starting data_frame position we will size from
        self._data_frame_start_tick = kwargs.get('data_frame_start_tick', 0)
        # how big the data window should be.
        self._data_frame_window = kwargs.get('data_frame_window', 500)
        # simply stores the length of the exchange df
        self._data_frame_length = self.environment.exchange.data_frame.shape[0]

        # Show a graph of the data frame window every generation
        self._graph_window = kwargs.get('graph_window', False)

        # output stats about each genomes performance every evaluation.
        self._watch_genome_evaluation = kwargs.get('watch_genome_evaluation', False)
        self._only_show_profitable = kwargs.get('only_show_profitable', False)

        # idk, this may be needed later
        self._sleep_between_evals = kwargs.get('sleep_between_evals', 0)

        # when pop.generation % this == 0 then the full data frame will be evaluated for every genome.
        # set to False to disable.
        self._full_evaluation_interval = kwargs.get('full_evaluation_interval', 20)
        # Force full evaluation
        self._full_evaluation = kwargs.get('full_evaluation', False)

        self._build_population()

    def load_config(self):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        self._neat_config_filename)
        config.genome_config.num_inputs = len(self._environment.exchange.data_frame.columns)
        config.genome_config.num_hidden = len(self._environment.exchange.data_frame.columns)
        config.genome_config.input_keys = [-i - 1 for i in range(config.genome_config.num_inputs)]
        return config

    @property
    def environment(self):
        return self._environment

    def _random_data_frame_start_tick(self):
        return random.randint(0, self._data_frame_length - self._data_frame_window)

    def _get_data_frame_window(self, start=None, advance=None, end=None):
        # find a random window to evaluate all genomes on
        if (self._pop.generation + 1) % self._full_evaluation_interval is 0 or self._full_evaluation is True:
            self.data_frame_start_tick = 0
            self._data_frame_window = self._data_frame_length-1
        else:
            self.data_frame_start_tick = self._random_data_frame_start_tick()

        if advance is None:
            advance = 0

        if start is None:
            start = self.data_frame_start_tick + advance

        if end is None:
            end = start + 1

        assert start < end, 'start timestep must be before end timestep'
        assert end < len(self._environment._exchange.data_frame), 'end time step out of bounds'

        return self._environment._exchange.data_frame[start:end]

    def _get_current_observation(self, advance=0):
        return self._get_data_frame_window(advance).values.flatten()

    @property
    def data_frame_start_tick(self):
        return self._data_frame_start_tick

    @data_frame_start_tick.setter
    def data_frame_start_tick(self, start=None):
        if start is not None:
             self._data_frame_start_tick = abs(int(start))
             return self._data_frame_start_tick
        else:
            return self._random_data_frame_start_tick()

    def _build_population(self):
        # create population
        self._pop = neat.Population(self._config)
        # add reporting
        self._pop.add_reporter(neat.StdOutReporter(True))
        self._stats = neat.StatisticsReporter()
        self._pop.add_reporter(self._stats)
        self._pop.add_reporter(neat.Checkpointer(5))


    def restore_agent(self, path: str, model_path: str = None):
        raise NotImplementedError

    def save_agent(self, path: str, model_path: str = None, append_timestep: bool = False):
        raise NotImplementedError

    def _finished_episode_cb(self) -> bool:
        raise NotImplementedError

    def tune(self, steps: int = None, episodes: int = None, callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        raise NotImplementedError

    def _derive_action(self, output):
        try:
            action = int(self._actions/2 * (1 + math.tanh(output[0])))
        except:
            print("*****ERROR IN DERIVE ACTION********", output, self._actions)
            action = -1
        return action

    def _report_genome_evaluation(self, genome):
        if self._watch_genome_evaluation:
            p = self._genome_performance[genome.key]

            if self._only_show_profitable is True and int(p['net_worth']) <= 10000:
                return

            print("Genome ID: ", genome.key)
            print("Rewards:", p['rewards'])
            print('Balance:', p['balance'])
            print("Net Worth:", p['net_worth'])
            print('Steps Completed', p['steps_completed'])
            print('Most common action', Counter(p['actions']))

            print('Number of trades:', Counter(self._environment.exchange.trades['type']))

        return

    def _do_graph_window(self):
        if self._graph_window:
            # show the current plot for the price window.
            plt.clf()
            window = self._get_data_frame_window()
            plt.plot(window['close'].values)
            plt.show()


    def _prep_eval(self):
        # find a random window to evaluate all genomes on
        if (self._pop.generation + 1) % self._full_evaluation_interval is 0 or self._full_evaluation is True:
            self.data_frame_start_tick = 0
            self._data_frame_window = self._data_frame_length-1
        else:
            self.data_frame_start_tick = self._random_data_frame_start_tick()

        self._do_graph_window()


    def _eval_population(self, genomes, config):
        self._prep_eval()
        for genome_id, genome in genomes:
            if not self._watch_genome_evaluation:
                print('*',end='')

            self._environment.reset()
            self._genome_performance[genome.key] = deepcopy(self._performance_stub)
            # set the current_step to the start of our window
            self.environment.exchange._current_step = self._data_frame_start_tick
            self.environment._current_step = self._data_frame_start_tick

            self.eval_genome(genome)

        clear_output()

    def _threaded_eval(self, genome, config)
        return

    def eval_genome(self, genome):
        if self._watch_genome_evaluation:
            print('---------------------------')

        # Initialize the network for this genome
        net = neat.nn.RecurrentNetwork.create(genome, self._config)
        # calculate the steps and keep track of some intial variables
        steps_completed = 0
        done = False
        starting_balance = self._environment.exchange.balance
        self._genome_performance[genome.key] = deepcopy(self._performance_stub)

        # set inital reward
        genome.fitness = 0.0

        # walk all timesteps to evaluate our genome
        # while (steps is not None and (steps == 0 or steps_completed < (steps))):
        while(steps_completed < self._data_frame_window):
            # activate() the genome and calculate the action output
            output = net.activate(self._get_current_observation(steps_completed))

            # action at current step
            action =  self._derive_action(output)
            if action is -1:
                print('BROKEN ACTION', output)
                genome.fitness = -100000
                break

            # feed action into environment to get reward for selected action
            obs, rewards, done, info = self.environment.step(action)

            # feed rewards to NEAT to calculate fitness.
            genome.fitness += rewards

            # count this as a completed step
            steps_completed += 1

            # stop iterating if we haven't learned to trade or we pass a fitness threshold
            if genome.fitness < self._learn_to_trade_theshold:
                if self._watch_genome_evaluation:
                    print("Learn to trade asshole!")
                done= True


            self._genome_performance[genome.key]['rewards'] += rewards
            self._genome_performance[genome.key]['actions'].append(action)
            self._genome_performance[genome.key]['steps_completed'] = steps_completed
            self._genome_performance[genome.key]['trades'] = len(self._environment.exchange.trades)
            self._genome_performance[genome.key]['balance'] = self._environment.exchange.balance
            self._genome_performance[genome.key]['net_worth'] = self._environment.exchange.net_worth

            if done:
                if self._watch_genome_evaluation:
                    print('-------WE DONE!---------')
                break

        # ballance our reward by how much profit we've made in our trading session.

        self._report_genome_evaluation(genome)
        return

    def run(self, generations: int = None, testing: bool = True, episode_callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        # Run for up to 300 generations.

        # pe = neat.ParallelEvaluator(10, self._eval)
        winner = self._pop.run(self._eval_population, generations)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        # print('\nOutput:')

        # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')

        return [self._environment._exchange.performance, winner, self._stats]
