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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        logger.info(f"\n{'='*80}\n开始处理币种: {pair}\n{'='*80}")
        logger.info(f"\n{'-'*40}\n{pair} - 开始计算技术指标\n{'-'*40}")

        # 1. 计算布林带
        bb = qtpylib.bollinger_bands(
            dataframe["close"],
            window=self.bb_period.value,
            stds=self.bb_std.value
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
                f"{pair} - 技术指标计算完成:\n"
                f"时间: {last.name}\n"
                f"收盘价: {last['close']:.4f}\n"
                f"布林上轨: {last['布林上轨']:.4f}\n"
                f"布林中轨: {last['布林中轨']:.4f}\n"
                f"布林下轨: {last['布林下轨']:.4f}\n"
                f"带宽: {last['带宽']:.4f}\n"
                f"成交量比例: {last['成交量比例']:.2f}"
            )
        logger.info(f"\n{'-'*40}\n{pair} - 技术指标计算完成\n{'-'*40}")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        logger.info(f"\n{'-'*40}\n{pair} - 开始计算入场信号\n{'-'*40}")

        # 1. 多头入场条件
        long_cond = (
            (dataframe["close"] > dataframe["布林上轨"]) &
            (dataframe["close"].shift(1) <= dataframe["布林上轨"].shift(1)) &
            (dataframe["成交量比例"] > 1.5)
        )

        # 2. 空头入场条件
        short_cond = (
            (dataframe["close"] < dataframe["布林下轨"]) &
            (dataframe["close"].shift(1) >= dataframe["布林下轨"].shift(1)) &
            (dataframe["成交量比例"] > 1.5)
        )

        # 3. 设置信号
        dataframe.loc[long_cond, ["enter_long", "enter_tag"]] = (1, "突破布林上轨")
        dataframe.loc[short_cond, ["enter_short", "enter_tag"]] = (1, "突破布林下轨")

        # 4. 输出信号统计
        if len(dataframe) > 0:
            long_count = long_cond.sum()
            short_count = short_cond.sum()
            logger.info(
                f"{pair} - 入场信号统计:\n"
                f"多头信号数: {long_count}\n"
                f"空头信号数: {short_count}"
            )
        logger.info(f"\n{'-'*40}\n{pair} - 入场信号计算完成\n{'-'*40}")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        logger.info(f"\n{'-'*40}\n{pair} - 开始计算出场信号\n{'-'*40}")

        # 1. 多头出场条件
        exit_long_cond = (
            (dataframe["close"] < dataframe["布林中轨"]) &
            (dataframe["close"].shift(1) >= dataframe["布林中轨"].shift(1))
        )

        # 2. 空头出场条件
        exit_short_cond = (
            (dataframe["close"] > dataframe["布林中轨"]) &
            (dataframe["close"].shift(1) <= dataframe["布林中轨"].shift(1))
        )

        # 3. 设置信号
        dataframe.loc[exit_long_cond, ["exit_long", "exit_tag"]] = (1, "跌破布林中轨")
        dataframe.loc[exit_short_cond, ["exit_short", "exit_tag"]] = (1, "突破布林中轨")

        # 4. 输出信号统计
        if len(dataframe) > 0:
            exit_long_count = exit_long_cond.sum()
            exit_short_count = exit_short_cond.sum()
            logger.info(
                f"{pair} - 出场信号统计:\n"
                f"平多信号数: {exit_long_count}\n"
                f"平空信号数: {exit_short_count}"
            )

        logger.info(f"\n{'-'*40}\n{pair} - 出场信号计算完成\n{'-'*40}")
        logger.info(f"\n{'='*80}\n{pair} - 处理完成\n{'='*80}")
        return dataframe
