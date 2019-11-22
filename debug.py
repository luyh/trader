import pandas as pd
data_file ='./data/processed/binance/btc_usdt_5m.csv'
df = pd.read_csv(data_file, index_col=[0])

import os,sys
if sys.platform == 'win32':
    parralle = True
    # number of days we want to pull from the dataframe
    days_of_data = 7
    pop_size = 10
    days = 1
    generations = 20
else:
    parralle = True
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

segments_in_day = 288

config = './neat.config'
strategy = TradingStrategy(environment=environment,
                           pop_size= pop_size,
                           initial_connectin='full_nodirect',
                           max_stagnation= 10,
                           neat_config=config,
                           watch_genome_evaluation=False,
                           only_show_profitable=True,
                           data_frame_window = segments_in_day * days,
                           disable_full_evaluation = True
                          )

if __name__ == '__main__':

    print("Running through ", strategy._data_frame_window, ' steps')
    # cp.run("performance, winner, stats = strategy.run(generations=20)", 'evolution_stats')
    performance, winner, stats = strategy.run(generations=generations,parralle = parralle)

    print('done')