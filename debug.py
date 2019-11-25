import pandas as pd
data_file ='./data/processed/binance/btc_usdt_5m.csv'
df = pd.read_csv(data_file, index_col=[0])

import os,sys
if sys.platform == 'win32':
    parralle = False
    num_workers = 2
    # number of days we want to pull from the dataframe
    days_of_data = 30
    pop_size = 50
    days = 1
    generations = 100
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
#from neat_stragtegy.neat_reward_strategy import NeatRewardStrategy as ProfitStrategy

from tensortrade.rewards import RiskAdjustedReturns as ProfitStrategy
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

import pickle
if __name__ == '__main__':

    strategy = TradingStrategy(environment=environment,
                               pop_size=pop_size,
                               initial_connectin='full_nodirect',
                               max_stagnation=10,
                               neat_config=config,
                               watch_genome_evaluation=True,
                               only_show_profitable=True,
                               data_frame_window=segments_in_day * days,
                               disable_full_evaluation=True
                               )

    print("Running through ", strategy._data_frame_window, ' steps')
    # cp.run("performance, winner, stats = strategy.run(generations=20)", 'evolution_stats')
    performance, winner, stats = strategy.run(generations=generations,
                                              parralle = parralle,
                                              num_workers = num_workers)

    with open("winners/winner-{}.bin".format(winner.key), "wb") as f:
        pickle.dump(winner, f, 2)

    exchange = Exchange(data_frame=df_test,
                        pretransform=True,
                        base_instrument='USDT',
                        commission_percent=0.075,
                        initial_balance=100,
                        window_size=1,
                        max_allowed_slippage_percent=3.0,
                        min_order_amount=1E-4,
                        min_trade_amount=1E-4,
                        observation_columns=df_test.columns
                        )
    environment = Environment(exchange=exchange,
                              action_scheme=action_scheme,
                              reward_scheme=reward_scheme,
                              feature_pipeline=feature_pipeline)

    environment.reset()
    # from neat_trading_strategy import NeatTradingStrategy as TradingStrategy
    # exchange.balance = 100000
    strategy = TradingStrategy(environment=environment,
                               neat_config=config,
                               watch_genome_evaluation=True,
                               only_show_profitable=True,
                               full_evaluation=True
                               )

    print("Starting Worth: {}".format(exchange.net_worth, exchange.balance))

    print("Running through ", strategy._data_frame_window, ' steps')
    strategy.eval_genome(winner, strategy._config)

    pd.set_option('display.max_colwidth', -1)
    exchange.performance.tail()
    exchange.trades.tail()

    import matplotlib.pyplot as plt

    ax = plt.subplot(211)
    exchange.performance.net_worth.plot(figsize=(15, 8), label='net worth', title='Net Worth')
    exchange.performance['com'] = exchange.trades.price * 0.00075
    ax2 = plt.subplot(212, sharex=ax)
    exchange.performance.com.plot(title='Commissions')
    plt.legend()
    plt.show()

    wins = 0
    losses = 0
    ax = plt.subplot(311)

    exchange._price_history.plot(figsize=(15, 8), label='actual price')
    print(df_test.columns)
    df_test.trend_ema_fast.plot(label='fast ema')
    df_test.trend_ema_slow.plot(label='slow ema')

    # print(exchange._price_history.head())
    # print(exchange.trades.head())
    # print(exchange.performance.head())
    for idx, trade in exchange.trades.iterrows():
        price_at_trade = exchange._price_history[trade.step - 1]
        if trade.type.is_buy:
            plt.plot(trade.step - 1, price_at_trade, "b^")
        elif trade.type.is_sell:
            the_buy_step = exchange.trades.iloc[idx - 1]
            profit = trade.price - the_buy_step.price
            if profit < 0:
                losses += 1
                color = '#d62728'
                plt.axvspan(the_buy_step.step - 1, trade.step - 1, alpha=0.5, facecolor='red')
                plt.plot(trade.step - 1, price_at_trade, "rD")

            elif profit >= 0:
                wins += 1
                color = '#2ca02c'
                plt.axvspan(the_buy_step.step - 1, trade.step - 1, alpha=0.5, facecolor='green')
                plt.plot(trade.step - 1, price_at_trade, "gD")

    adx = plt.subplot(312, sharex=ax)
    df_test.adx.plot(label='adx')
    df_test.adx_pos.plot(label='pos')
    df_test.adx_neg.plot(label='neg')
    df_test.adx_long.plot(label='long')

    df_test.momentum_rsi.plot(label='rsi')

    ax2 = plt.subplot(313, sharex=adx)
    exchange.performance.net_worth.plot(title='Net Worth')
    ax.legend()

    ax.set_title("Wins:{} Losses:{} Win rate {}%".format(wins, losses, round((wins / (wins + losses)) * 100, 4)))
    plt.show()

    print('done')

    print('done')