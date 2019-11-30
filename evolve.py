import pandas as pd
data_file ='./data/processed/binance/btc_usdt_5m.csv'
df = pd.read_csv(data_file, index_col=[0])[['close','adx']]

# number of data frames (our DF is in 1h timesteps)
frames = 7 * 24 * 12 # A WEEK
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

from tensortrade.rewards import RiskAdjustedReturns as ProfitStrategy
from tensortrade.actions import DiscreteActions
from tensortrade.exchanges.simulated import SimulatedExchange as Exchange
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features import FeaturePipeline
from tensortrade.environments import TradingEnvironment as Environment
print('fin imports')

close_normalize = MinMaxNormalizer(inplace=True,
                             input_min = 7300,
                             input_max = 8300,
                             feature_min = -1,
                             feature_max = 1,
                             columns=['close',])
adx_normalize = MinMaxNormalizer(inplace=True,
                             input_min = 0,
                             input_max = 100,
                             feature_min = -1,
                             feature_max = 1,
                             columns=['adx',])

feature_pipeline = FeaturePipeline(steps=[close_normalize,adx_normalize])

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

print(exchange.data_frame.head())

import neat
import visualize
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import time,os
import math


def eval_genome(env,genome, config):
    # Initialize the network for this genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    # calculate the steps and keep track of some intial variables
    steps_completed = 0
    fitness = 0.0
    obs = env.reset()

    while (steps_completed < 288):
        outputs = net.activate( obs )
        action = np.argmax(outputs)
        #print(obs,outputs,action)
        print(net.values,action)
        # feed action into environment to get reward for selected action
        obs, rewards, done, info = environment.step( action )
        fitness += rewards
        steps_completed += 1

        if done:
            break

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
                eval_genome(self.env,genome, config)
        else:
            jobs = []
            for genome_id, genome in genomes:
                jobs.append( self.pool.apply_async(eval_genome, (self.env,genome, config) ) )

            # assign the fitness back to each genome
            for job, (ignored_genome_id, genome) in zip( jobs, genomes ):
                genome.fitness = job.get( timeout=self.timeout )

        print("final fitness compute time {0}\n".format(time.time() - t0))


def run(NUM_CORES,GENERATIONS):
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

    gen_best = pop.run( ec.evaluate_genomes, GENERATIONS )

if __name__ == '__main__':
    GENERATIONS = 10
    NUM_CORES = 1
    run(NUM_CORES,GENERATIONS)