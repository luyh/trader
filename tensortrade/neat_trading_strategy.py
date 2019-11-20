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
from typing import Union, Callable, List, Dict
from IPython.display import clear_output
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt
from copy import deepcopy

import math
import random

import neat

from tensortrade.environments.trading_environment import TradingEnvironment
from tensortrade.features.feature_pipeline import FeaturePipeline
from tensortrade.strategies import TradingStrategy
from tensortrade.trades import TradeType




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
        self._actions = self._environment.action_scheme.n_actions

        # population controls
        self._pop_size = kwargs.get('pop_size', 20)
        self._max_stagnation = kwargs.get('max_stagnation', 15)
        self._species_elitism = kwargs.get('species_elitism', 3)
        self._elitism = kwargs.get('elitism', 2)
        self._survival_threshold = kwargs.get('survival_threshold', 0.2)
        self._min_species_size = kwargs.get('min_species_size', 2)

        # species controls
        self._compatibility_threshold = kwargs.get('compatibility_threshold', 5)

        # network controls
        self._feed_foward = kwargs.get('feed_forward', False)
        self._initial_connection = kwargs.get('initial_connection', 'full_direct')

        # connection controls
        self._enabled_default = kwargs.get('enabled_default', True)
        self._enabled_mutate_rate = kwargs.get('enabled_mutate_rate', 0.1)
        self._conn_add_prob = kwargs.get('conn_add_prob', 0.5)
        self._conn_delete_prob = kwargs.get('conn_delete_prob', 0.1)
        self._node_add_prob = kwargs.get('node_add_prob', 0.5)
        self._node_delete_prob = kwargs.get('node_delete_prob', 0.1)


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
        self._disable_full_evaluation = kwargs.get('disable_full_evaluation', False)
        # Force full evaluation
        self._full_evaluation = kwargs.get('full_evaluation', False)

        self._build_population()

    def load_config(self):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            self._neat_config_filename)

        print(config)
        config.genome_config.num_inputs = len(self._environment.exchange.data_frame.columns)
        config.genome_config.num_hidden = len(self._environment.exchange.data_frame.columns)
        config.genome_config.input_keys = [-i - 1 for i in range(config.genome_config.num_inputs)]
        config.genome_config.feed_forward = self._feed_foward
        config.pop_size = self._pop_size

        # config.species_set_config.species_fitness_func = self._species_fitness_func #TODO
        config.stagnation_config.max_stagnation = self._max_stagnation
        config.stagnation_config.species_elitism = self._species_elitism

        config.reproduction_config.elitism = self._elitism
        config.reproduction_config.survival_threshold = self._survival_threshold
        config.reproduction_config.min_species_size = self._min_species_size

        config.species_set_config.compatibility_threshold = self._compatibility_threshold

        config.genome_config.enabled_default = self._enabled_default
        config.genome_config.initial_connection = self._initial_connection
        config.genome_config.enabled_mutate_rate = self._enabled_mutate_rate
        config.genome_config.conn_add_prob = self._conn_add_prob
        config.genome_config.conn_delete_prob = self._conn_delete_prob
        config.genome_config.node_add_prob = self._node_add_prob
        config.genome_config.node_delete_prob = self._node_delete_prob

        return config

    @property
    def environment(self):
        return self._environment

    def _random_data_frame_start_tick(self):
        return random.randint(0, self._data_frame_length - self._data_frame_window)

    def _get_data_frame_window(self, start=None, advance=None, end=None):
        # find a random window to evaluate all genomes on
        if self._disable_full_evaluation is False and (self._pop.generation + 1) % self._full_evaluation_interval is 0 or self._full_evaluation is True:
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
        print(output[0])
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

            actions = defaultdict(int)
            for action, count in Counter(p['actions']).items():
                actions[self.environment.action_scheme._get_trade_type(action).name] = count

            print('Most common action', actions.items())

            print('Number of trades:', Counter(self.environment.exchange.trades['type']))

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
        if self._disable_full_evaluation is False and (self._pop.generation + 1) % self._full_evaluation_interval is 0 or self._full_evaluation is True:
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

            genome.fitness = self.eval_genome(genome, config)

        clear_output()

    def _threaded_eval(self, genome, config):
        self._prep_eval()
        genome.fitness = self.eval_genome(genome, config)
        return genome.fitness

    def eval_genome(self, genome, config: neat.Config = None):
        if self._watch_genome_evaluation:
            print('---------------------------')

        if config is None:
            config = self._config.copy()
        # Initialize the network for this genome
        net = neat.nn.RecurrentNetwork.create(genome, config)
        # calculate the steps and keep track of some intial variables
        steps_completed = 0
        done = False
        starting_balance = self._environment.exchange.balance
        self._genome_performance[genome.key] = deepcopy(self._performance_stub)

        # set inital reward
        fitness = 0.0

        # walk all timesteps to evaluate our genome
        # while (steps is not None and (steps == 0 or steps_completed < (steps))):
        while(steps_completed < self._data_frame_window):
            # activate() the genome and calculate the action output
            output = net.activate(self._get_current_observation(steps_completed))

            # action at current step
            action =  self._derive_action(output)
            if action is -1:
                print('BROKEN ACTION', output)
                fitness = -100000
                break

            # feed action into environment to get reward for selected action
            obs, rewards, done, info = self.environment.step(action)

            # feed rewards to NEAT to calculate fitness.
            fitness += rewards

            # count this as a completed step
            steps_completed += 1

            # stop iterating if we haven't learned to trade or we pass a fitness threshold
            if fitness < self._learn_to_trade_theshold:
                if self._watch_genome_evaluation:
                    print("Learn to trade asshole!")
                done= True


            self._genome_performance[genome.key]['rewards'] += rewards
            self._genome_performance[genome.key]['actions'].append(action)
            self._genome_performance[genome.key]['steps_completed'] = steps_completed
            self._genome_performance[genome.key]['trades'] = len(self.environment.exchange.trades)
            self._genome_performance[genome.key]['balance'] = self.environment.exchange.balance
            self._genome_performance[genome.key]['net_worth'] = self.environment.exchange.net_worth

            if done:
                if self._watch_genome_evaluation:
                    print('-------WE DONE!---------')
                break

        # ballance our reward by how much profit we've made in our trading session.

        self._report_genome_evaluation(genome)
        return fitness

    def run(self, generations: int = None, testing: bool = True, episode_callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        # Run for up to 300 generations.

        pe = neat.ParallelEvaluator(10, self._threaded_eval)
        winner = self._pop.run(pe.evaluate, generations)
        # winner = self._pop.run(self._eval_population, generations)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        # print('\nOutput:')

        # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')

        return [self._environment._exchange.performance, winner, self._stats]
