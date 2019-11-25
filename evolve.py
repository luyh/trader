import pandas as pd
data_file ='../data/processed/binance/btc_usdt_5m.csv'
df = pd.read_csv(data_file, index_col=[0])

import os,sys
if sys.platform == 'win32' or sys.platform == 'darwin':
    parralle = True
    num_workers = 2
    # number of days we want to pull from the dataframe
    days_of_data = 7
    pop_size = 10
    days = 1
    generations = 20
else:
    parralle = False
    num_workers = 10
    days_of_data = 365
    pop_size = 50
    days = 7
    generations = 100

print('days_of_data={},pop_size = {},days = {},generations = {},'.format( days_of_data,pop_size ,days,generations))
# number of data frames (our DF is in 1h timesteps)
frames = days_of_data * 24 * 12
# frames = len(df)
train_test_percentage = 0.4

x_train = int(frames * (1 - train_test_percentage))
x_test = int(frames - x_train)

df_train = df[-frames:(-x_test - 1)]
df_test = df[-x_test:]
print("Friend Shape", df.shape)
print('train shape', df_train.shape)
print('test shape', df_test.shape)
print('columns', df.columns)

print(df_test.head())

del df

import tensortrade
from neat_stragtegy.neat_trading_strategy import NeatTradingStrategy as TradingStrategy
from neat_stragtegy.neat_reward_strategy import NeatRewardStrategy as ProfitStrategy

from tensortrade.actions import DiscreteActions
from tensortrade.exchanges.simulated import SimulatedExchange as Exchange
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features import FeaturePipeline
from tensortrade.environments import TradingEnvironment as Environment
from tensortrade.trades import Trade

print('fin imports')
normalize = MinMaxNormalizer(inplace=True,
                             input_min = 2000,
                             input_max = 20000)
feature_pipeline = FeaturePipeline(steps=[normalize])

reward_scheme = ProfitStrategy()
action_scheme = DiscreteActions(n_actions=5, instrument='BTC/USDT')
print('fin strats')

exchange = Exchange(data_frame=df_train,
                    pretransform = True,
                    base_instrument='USDT',
                    commission_percent=0.075,
                    initial_balance = 100,
                    window_size=1,
                    max_allowed_slippage_percent=3.0,
                    min_order_amount=1E-4,
                    min_trade_amount=1E-4,
                    observation_columns = df_train.columns
                   )

print('fin exchange')

environment = Environment(exchange=exchange,
                                 action_scheme=action_scheme,
                                 reward_scheme=reward_scheme,
                                 feature_pipeline=feature_pipeline)

print('fin environment')
print('')

environment.exchange._balance = 200

print(exchange.balance,environment.exchange._balance)

import neat
import visualize
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import time,os
import math

def derive_action(output):
    # print(output[0])
    try:
        action = int( self._actions / 2 * (1 + math.tanh( output[0] )) )
    except:
        print( "*****ERROR IN DERIVE ACTION********", output, self._actions )
        action = -1
    return action

def eval_genome(env,genome, config: neat.Config = None):
    # Initialize the network for this genome
    net = neat.nn.RecurrentNetwork.create( genome, config )
    # calculate the steps and keep track of some intial variables
    steps_completed = 0
    done = False

    # set inital reward
    fitness = 0.0
    obs = env._next_observation( Trade( 'N/A', 'hold', 0, 0 ) )
    # walk all timesteps to evaluate our genome
    # while (steps is not None and (steps == 0 or steps_completed < (steps))):
    while (steps_completed < 288):
        # print('steps_completed:{},_data_frame_window:{}'.format(steps_completed,self._data_frame_window))
        # activate() the genome and calculate the action output

        output = net.activate( obs )

        # action at current step
        action = derive_action( output )
        if action is -1:
            print( 'BROKEN ACTION', output )
            fitness = -100000
            break

        # feed action into environment to get reward for selected action
        obs, rewards, done, info = self.environment.step( action )

        # feed rewards to NEAT to calculate fitness.
        fitness += rewards

        # count this as a completed step
        steps_completed += 1

        # stop iterating if we haven't learned to trade or we pass a fitness threshold
        if fitness < self._learn_to_trade_theshold:
            if self._watch_genome_evaluation:
                print( "Learn to trade asshole!" )
            done = True

        self._genome_performance[genome.key]['rewards'] += rewards
        self._genome_performance[genome.key]['actions'].append( action )
        self._genome_performance[genome.key]['steps_completed'] = steps_completed
        self._genome_performance[genome.key]['trades'] = len( self.environment.exchange.trades )
        self._genome_performance[genome.key]['balance'] = self.environment.exchange.balance
        self._genome_performance[genome.key]['net_worth'] = self.environment.exchange.net_worth

        if done:
            if self._watch_genome_evaluation:
                print( '-------WE DONE!---------' )
            break

    # ballance our reward by how much profit we've made in our trading session.

    self._report_genome_evaluation( genome )
    return fitness


class PooledEvaluate(object):
    def __init__(self,NUM_CORES, env ,timeout = None):
        self.timeout = timeout
        self.pool = None if NUM_CORES < 2 else multiprocessing.Pool(NUM_CORES)
        self.env = env


    def evaluate_genomes(self, genomes, config):
        t0 = time.time()

        # Assign a composite fitness to each genome; genomes can make progress either
        # by improving their total reward or by making more accurate reward estimates.
        # print("Evaluating {0} test episodes".format(len(self.test_episodes)))
        if self.pool is None:
            for genome_id, genome in genomes:
                eval_genome(genome, config)
        else:
            jobs = []
            for genome_id, genome in genomes:
                jobs.append( self.pool.apply_async(eval_genome, (self.env,genome, config) ) )

            # assign the fitness back to each genome
            for job, (ignored_genome_id, genome) in zip( jobs, genomes ):
                genome.fitness = job.get( timeout=self.timeout )

        print("final fitness compute time {0}\n".format(time.time() - t0))


def run(NUM_CORES):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    pop.add_reporter(neat.Checkpointer(5))

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    ec = PooledEvaluate(NUM_CORES= NUM_CORES,env = environment)

    gen_best = pop.run( ec.evaluate_genomes, 5 )

if __name__ == '__main__':
    NUM_CORES = 2
    run(NUM_CORES)