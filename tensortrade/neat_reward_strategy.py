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
from tensortrade.rewards import RewardStrategy

import pandas as pd
import numpy as np

from abc import ABCMeta, abstractmethod

from tensortrade.trades import Trade


class NeatRewardStrategy(RewardStrategy, metaclass=ABCMeta):

    def __init__(self):
        self._purchase_price = -1
        self._is_holding_instrument = False
        self.phi = (1 + 5 ** 0.5) / 2
        pass

    def reset(self):
        """Necessary to reset the last purchase price and state of open positions."""
        self._purchase_price = -1
        self._is_holding_instrument = False

    def get_reward(self, current_step: int, trade: Trade) -> float:
        """Reward -1 for not holding a position, 1 for holding a position, 2 for opening a position, and 1 + 5^(log_10(profit)) for closing a position.

        The 5^(log_10(profit)) function simply slows the growth of the reward as trades get large.
        """
        reward = 0
        # I am holding an instrument, I will be rewarded if the price is moving positivly
        if trade.is_hold and self._is_holding_instrument:
            profit_per_instrument = exchange.current_price - self._purchase_price
            profit = trade.amount * profit_per_instrument
            print('holding', profit, exchange.trades)
            profit_sign = np.sign(profit)
            reward = profit_sign * (1 + (np.log(abs(profit))))

        elif trade.is_hold and not self._is_holding_instrument and current_step > 5:
            # I am NOT holding an instrument and I am holding...
            # I will be rewarded if holding was the right decision.

            # positive or negative increase?
            d = self.exchange.data_frame[current_step-5:current_step]['close'].values

            #calculate % increase or decrease
            pos_or_neg = np.sign(100 * ((d[-1] - d[0]) / d[0]))
            r = 0
            if pos_or_neg > 0:
                # if the price has rizen in the previous 5 time steps we have missed an oportunity and should be penalized.
                # we will penalize by a log of the profit we would have made had we bought and sold at this timestep
                r = -1*((self.phi ** np.log10(abs(d[0] - d[-1]))))
            elif pos_or_neg < 0:
                # if the price has dropped in the previous 5 time steps then we have correctly held and should be rewarded
                # we will reward by a log of the inverse of the loss, because we have saved money.
                r = abs((self.phi ** np.log10(abs(d[-1] - d[0]))))
            reward= r

        elif trade.is_buy and trade.amount > 0 and trade.price < self.exchange.balance:
            self._purchase_price = trade.price
            self._is_holding_instrument = True
            reward= 1

        elif trade.is_sell and trade.amount > 0:
            self._is_holding_instrument = False
            profit_per_instrument = trade.price - self._purchase_price
            profit = trade.amount * profit_per_instrument
            profit_sign = np.sign(profit)
            reward = profit_sign * (1 + ((2**self.phi) ** np.log10(abs(profit))))

        if reward is 0:
            print(current_step, reward, self.exchange.balance, self.exchange.net_worth, trade._amount, trade._price, trade._trade_type)
        return reward
