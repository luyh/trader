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
# limitations under the License
from tensortrade.rewards import RewardScheme

import pandas as pd
import numpy as np
from math import log
from abc import ABCMeta, abstractmethod

from tensortrade.trades import Trade


class NeatRewardStrategy(RewardScheme, metaclass=ABCMeta):

    def __init__(self):
        self._purchase_price = 0
        self._is_holding_instrument = False
        self.phi = (1 + 5 ** 0.5) / 2
        pass

    def reset(self):
        """Necessary to reset the last purchase price and state of open positions."""
        self._purchase_price = -1
        self._is_holding_instrument = False

    def get_reward(self, current_step: int, trade: Trade) -> float:
        reward = 0
        current_price = self.exchange.current_price(trade.symbol)
        profit_per_instrument = current_price - self._purchase_price
        abs_profit = abs(trade.transact_amount * profit_per_instrument)
        sign = np.sign(trade.transact_amount * profit_per_instrument)

        # net Worth
        # net_worth_change = 10000 - self.exchange.net_worth


        if trade.valid:
            if trade.is_hold and self._is_holding_instrument:
                reward = 1*sign + log(1 + abs_profit)
                self._is_holding_instrument = True
            elif trade.is_buy and not self._is_holding_instrument:
                # did we just sell and buy on an upswing?
                self._is_holding_instrument = True
                reward = 1
            elif trade.is_sell and self._is_holding_instrument:
                reward = 1*sign + log(1 + abs_profit)
                self.reset()
        else:
            reward = 0

        return reward

    # def get_reward(self, current_step: int, trade: Trade) -> float:
    #     """Reward -1 for not holding a position, 1 for holding a position, 2 for opening a position, and 1 + 5^(log_10(profit)) for closing a position.
    #
    #     The 5^(log_10(profit)) function simply slows the growth of the reward as trades get large.
    #     """
    #     current_price = self.exchange.current_price(trade.symbol)
    #     reward = 0
    #     # I am holding an instrument, I will be rewarded if the price is moving positivly
    #     if not trade.valid:
    #         reward = -1
    #         # print("reward: Trade is not valid : {}".format(trade.to_dict))
    #         if trade.is_hold and not self._is_holding_instrument:
    #             # I am NOT holding an instrument and I am holding...
    #             # I will be rewarded if holding was the right decision.
    #
    #             #calculate % increase or decrease
    #             price_diff = self._purchase_price - current_price
    #             pos_or_neg = np.sign(100 * ((price_diff) / current_price))
    #
    #             return 1 + np.log(1 + (current_price - self._purchase_price))
    #
    #             if pos_or_neg > 0:
    #                 # if the price has rizen in the previous 5 time steps we have missed an oportunity and should be penalized.
    #                 # we will penalize by a log of the profit we would have made had we bought and sold at this timestep
    #                 reward = -1*((self.phi ** np.log10(price_diff)))
    #
    #             elif pos_or_neg < 0:
    #                 # if the price has dropped in the previous 5 time steps then we have correctly held and should be rewarded
    #                 # we will reward by a log of the inverse of the loss, because we have saved money.
    #                 reward = abs((self.phi ** np.log10(abs(price_diff))))
    #
    #         return reward
    #     else:
    #         return 1 + np.log(1 + (current_price - self._purchase_price))
    #
    #     if trade.is_hold and self._is_holding_instrument:
    #         profit_per_instrument = current_price - self._purchase_price
    #         profit = trade.transact_amount * profit_per_instrument
    #         # print('holding', profit, exchange.trades)
    #         profit_sign = np.sign(profit)
    #         reward = profit_sign * (1 + (np.log(abs(profit))))
    #
    #     elif trade.is_buy:
    #         reward= 10
    #         d = self.exchange.data_frame[current_step-1:current_step+1]['close'].values
    #
    #         # did you buy too early?
    #         # get next timestep price and compare to current price, deduct points if price in future is lower
    #         # did you buy too late?
    #         # get previous timestep price and compare to current price, deduct points if price in past is lower
    #         if int(d[0]) is 0 or int(d[2]) is 0:
    #             reward=10
    #         elif d[0]<current_price:
    #             reward = reward - (current_price - d[0])
    #         elif d[2]<current_price:
    #             reward = reward - (current_price - d[2])
    #
    #         self._purchase_price = trade.transact_price
    #         self._is_holding_instrument = True
    #
    #     elif trade.is_sell:
    #         self._is_holding_instrument = False
    #         self._purchase_price = 0
    #         profit_per_instrument = trade.transact_price - self._purchase_price
    #         profit = trade.transact_amount * profit_per_instrument
    #         profit_sign = np.sign(profit)
    #         reward = profit_sign * (1 + ((2**self.phi) ** np.log10(abs(profit))))
    #
    #     # if reward is 0:
    #     #     print("0 reward", current_step, reward, self.exchange.balance, self.exchange.net_worth, trade.transact_amount, trade.transact_price, trade._trade_type)
    #
    #     # print('reward', reward)
    #     return reward
