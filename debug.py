import tensorflow as tf
tf.enable_eager_execution()
tf.executing_eagerly()

import warnings
import numpy
def warn(*args, **kwargs):
    pass

warnings.warn = warn
warnings.simplefilter(action='ignore', category=FutureWarning)
numpy.seterr(divide = 'ignore')

import sys,os
sys.path.append(os.path.dirname(os.path.abspath('')))
sys.path.append(os.path.abspath('')+"\\neat_stragtegy")

import pandas as pd
data_file ='./data/processed/binance/btc_usdt_1h.csv'
df = pd.read_csv(data_file, index_col=[0])


# number of days we want to pull from the dataframe
days_of_data = 365

# number of data frames (our DF is in 1h timesteps)
frames = days_of_data * 24 * 5
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

import tensortrade
from neat_stragtegy.neat_trading_strategy import NeatTradingStrategy as TradingStrategy
from neat_stragtegy.neat_reward_strategy import NeatRewardStrategy as ProfitStrategy
from tensortrade.actions import DiscreteActions
from tensortrade.exchanges.simulated import SimulatedExchange as Exchange
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features import FeaturePipeline
from tensortrade.environments import TradingEnvironment as Environment

print('fin imports')
normalize = MinMaxNormalizer(inplace=True)
feature_pipeline = FeaturePipeline(steps=[normalize])

reward_scheme = ProfitStrategy()
action_scheme = DiscreteActions(n_actions=5, instrument='BTC/USDT')
print('fin strats')

exchange = Exchange(data_frame=df_train,
                    pretransform = True,
                    base_instrument='USDT',
                    commission_percent=0.75,
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

segments_in_day = 288
days = 7

config = './neat.config'
strategy = TradingStrategy(environment=environment,
                           neat_config=config,
                           watch_genome_evaluation=False,
                           only_show_profitable=False,
                           learn_to_trade_theshold=-10000,
                           data_frame_window = segments_in_day * days
                          )

print("Running through ", strategy._data_frame_window, ' steps')
# cp.run("performance, winner, stats = strategy.run(generations=20)", 'evolution_stats')
performance, winner, stats = strategy.run(generations=200)

print('done')