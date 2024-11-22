import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timezone
from freqtrade.strategy import IStrategy, IntParameter
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging

logger = logging.getLogger(__name__)


class BBBreakoutStrategy(IStrategy):
    """
    自适应布林带策略
    作者: Claude
    版本: 1.0
    说明: 用于学习Freqtrade的策略开发流程
    """

    INTERFACE_VERSION = 3
    timeframe = "1m"  # 使用1分钟K线便于观察

    # 止盈设置
    minimal_roi = {
        "0": 0.05,  # 立即止盈点位
        "30": 0.03,  # 30分钟后的止盈点
        "60": 0.01,  # 60分钟后的止盈点
        "120": 0,  # 120分钟后的止盈点
    }

    # 止损设置
    stoploss = -0.05  # 止损比例
    trailing_stop = True  # 启用追踪止损
    trailing_stop_positive = 0.01  # 盈利后的追踪止损
    trailing_stop_positive_offset = 0.02  # 追踪止损偏移

    # 布林带参数
    bb_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    bb_std = IntParameter(1, 3, default=2, space="buy", optimize=True)

    plot_config = {
        "main_plot": {
            "BB布林上轨": {"color": "red"},
            "BB布林中轨": {"color": "blue"},
            "BB布林下轨": {"color": "red"},
        },
        "subplots": {
            "指标": {
                "带宽": {"color": "yellow"},
                "成交量比例": {"color": "orange"},
                "title": "Indicators",
            }
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        logger.info(f"\n{'='*80}\n{pair} - 开始本币流程\n{'='*80}")

        # 1. 计算布林带
        bb = qtpylib.bollinger_bands(
            dataframe["close"], window=self.bb_period.value, stds=self.bb_std.value
        )
        logger.info(f"{pair} - 布林带参数: 周期={self.bb_period.value}, 倍数={self.bb_std.value}")

        # 2. 基础指标赋值
        dataframe["布林上轨"] = bb["upper"]
        dataframe["布林中轨"] = bb["mid"]
        dataframe["布林下轨"] = bb["lower"]

        # 3. 计算布林带宽度
        dataframe["带宽"] = (bb["upper"] - bb["lower"]) / bb["mid"]

        # 4. 计算成交量指标
        dataframe["成交量均值"] = dataframe["volume"].rolling(window=20).mean()
        dataframe["成交量比例"] = dataframe["volume"] / dataframe["成交量均值"]

        # 5. 输出计算结果
        if len(dataframe) > 0:
            last = dataframe.iloc[-1]
            logger.info(
                f"{pair} - 计算指标: 收盘价={last['close']:.4f}, 中轨={last['布林中轨']:.4f}, "
                f"上轨={last['布林上轨']:.4f}, 下轨={last['布林下轨']:.4f}, "
                f"带宽={last['带宽']:.4f}, 成交量比={last['成交量比例']:.2f}"
            )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        logger.info(f"{pair} - 开始计算入场信号")

        # 1. 多头入场条件
        long_cond = (
            (dataframe["close"] > dataframe["布林上轨"])
            & (dataframe["close"].shift(1) <= dataframe["布林上轨"].shift(1))
            & (dataframe["成交量比例"] > 1.5)
        )

        # 2. 空头入场条件
        short_cond = (
            (dataframe["close"] < dataframe["布林下轨"])
            & (dataframe["close"].shift(1) >= dataframe["布林下轨"].shift(1))
            & (dataframe["成交量比例"] > 1.5)
        )

        # 3. 设置信号
        dataframe.loc[long_cond, ["enter_long", "enter_tag"]] = (1, "突破布林上轨")
        dataframe.loc[short_cond, ["enter_short", "enter_tag"]] = (1, "突破布林下轨")

        # 4. 输出最新K线的信号值
        if len(dataframe) > 0:
            if long_cond.iloc[-1] or short_cond.iloc[-1]:
                logger.info(
                    f"{metadata['pair']} - 计算开仓信号: "
                    f"多头={1 if long_cond.iloc[-1] else 0}, "
                    f"空头={1 if short_cond.iloc[-1] else 0}"
                )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        logger.info(f"{pair} - 开始计算出场信号")

        # 1. 多头出场条件
        exit_long_cond = (dataframe["close"] < dataframe["布林中轨"]) & (
            dataframe["close"].shift(1) >= dataframe["布林中轨"].shift(1)
        )

        # 2. 空头出场条件
        exit_short_cond = (dataframe["close"] > dataframe["布林中轨"]) & (
            dataframe["close"].shift(1) <= dataframe["布林中轨"].shift(1)
        )

        # 3. 设置信号
        dataframe.loc[exit_long_cond, ["exit_long", "exit_tag"]] = (1, "跌破布林中轨")
        dataframe.loc[exit_short_cond, ["exit_short", "exit_tag"]] = (1, "突破布林中轨")

        # 4. 只输出平仓信号
        if len(dataframe) > 0:
            if exit_long_cond.iloc[-1] or exit_short_cond.iloc[-1]:
                logger.info(
                    f"{metadata['pair']} - 计算平仓信号: "
                    f"平多={1 if exit_long_cond.iloc[-1] else 0}, "
                    f"平空={1 if exit_short_cond.iloc[-1] else 0}"
                )
        return dataframe
