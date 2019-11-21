import pandas as pd
data_file ='./data/processed/binance/btc_usdt_5m.csv'
df = pd.read_csv(data_file, index_col=[0])

import pickle
winner = "winner-1601.bin"
with open("winners/{}".format(winner), "rb") as f:
    winner = pickle.load(f)

days_of_data = 365
pop_size = 500
days = 7
generations = 100
config = './neat.config'


frames = days_of_data * 24 * 5
train_test_percentage = 0.4
x_train = int(frames * (1 - train_test_percentage))
x_test = int(frames - x_train)
df_test = df[-x_test:]

print(df_test.shape)
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

exchange = Exchange(data_frame=df_test,
                    pretransform = True,
                    base_instrument='USDT',
                    commission_percent=0.075,
                    window_size=1,
                    max_allowed_slippage_percent=3.0,
                    min_order_amount=1E-4,
                    min_trade_amount=1E-4,
                    observation_columns = df_test.columns
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
                           only_show_profitable=False,
                           full_evaluation=True
                          )

exchange._balance = 100
print("Starting Worth: {}".format(exchange.net_worth, exchange.balance))

print("Running through ", strategy._data_frame_window, ' steps')
strategy.eval_genome(winner, strategy._config)

pd.set_option('display.max_colwidth', -1)
exchange.performance.tail()
exchange.trades.tail()

import matplotlib.pyplot as plt
ax = plt.subplot(211)
exchange.performance.net_worth.plot(figsize=(15,8), label='net worth', title='Net Worth')
exchange.performance['com'] = exchange.trades.price*0.00075
ax2 = plt.subplot(212, sharex=ax)
exchange.performance.com.plot(title='Commissions')
plt.legend()
plt.show()


wins = 0
losses = 0
ax = plt.subplot(311)

exchange._price_history.plot(figsize=(15,8), label='actual price')
print(df_test.columns)
df_test.trend_ema_fast.plot(label='fast ema')
df_test.trend_ema_slow.plot(label='slow ema')

# print(exchange._price_history.head())
# print(exchange.trades.head())
# print(exchange.performance.head())
for idx,trade in exchange.trades.iterrows():
    price_at_trade = exchange._price_history[trade.step-1]
    if trade.type.is_buy:
        plt.plot(trade.step-1 , price_at_trade, "b^")
    elif trade.type.is_sell:
        the_buy_step = exchange.trades.iloc[idx-1]
        profit = trade.price - the_buy_step.price
        if profit < 0:
            losses +=1
            color = '#d62728'
            plt.axvspan(the_buy_step.step-1, trade.step-1, alpha= 0.5, facecolor='red')
            plt.plot(trade.step-1, price_at_trade, "rD")

        elif profit >= 0:
            wins +=1
            color = '#2ca02c'
            plt.axvspan(the_buy_step.step-1, trade.step-1, alpha= 0.5, facecolor='green')
            plt.plot(trade.step-1, price_at_trade, "gD")

adx = plt.subplot(312, sharex=ax)
df_test.adx.plot(label='adx')
df_test.adx_pos.plot(label='pos')
df_test.adx_neg.plot(label='neg')
df_test.adx_long.plot(label='long')

df_test.momentum_rsi.plot(label='rsi')

ax2 = plt.subplot(313, sharex=adx)
exchange.performance.net_worth.plot(title='Net Worth')
ax.legend()

ax.set_title("Wins:{} Losses:{} Win rate {}%".format(wins, losses, round((wins/(wins+losses))*100, 4) ))
plt.show()

print('done')