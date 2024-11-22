import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timezone
from freqtrade.strategy import IStrategy, IntParameter
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging

logger = logging.getLogger(__name__)


class AdaptiveBollingerStrategy(IStrategy):
    """
    自适应布林带策略
    作者: Claude
    版本: 1.0
    说明: 使用自适应标准差倍数的布林带策略
    特点:
    1. 布林带宽度会根据市场波动自动调整
    2. 在波动剧烈时期自动加宽带宽
    3. 在波动平缓时期自动收窄带宽
    """

    INTERFACE_VERSION = 3
    timeframe = "1m"

    # 止盈设置
    minimal_roi = {"0": 0.05, "30": 0.03, "60": 0.01, "120": 0}

    # 止损设置
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    # 布林带参数
    bb_period = IntParameter(20, 50, default=40, space="buy", optimize=True)

    # 添加绘图配置
    plot_config = {
        "main_plot": {
            # 主图显示自适应布林带
            "自适应布林上轨": {"color": "red"},
            "自适应布林中轨": {"color": "blue"},
            "自适应布林下轨": {"color": "red"},
        },
        "subplots": {
            "Z分数": {"Z分数": {"color": "yellow"}, "title": "Z-Score"},
            "动态倍数": {"动态倍数": {"color": "orange"}, "title": "Dynamic Multiplier"},
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        logger.info(f"\n{'='*80}\n{pair} - 开始本币流程\n{'='*80}")

        # 1. 计算基础指标
        n = self.bb_period.value
        dataframe["中轨"] = dataframe["close"].rolling(n, min_periods=1).mean()
        dataframe["标准差"] = dataframe["close"].rolling(n, min_periods=1).std(ddof=0)

        # 2. 计算z-score(标准化分数)
        dataframe["Z分数"] = abs(dataframe["close"] - dataframe["中轨"]) / dataframe["标准差"]

        # 3. 计算动态倍数(历史最大Z分数)
        dataframe["动态倍数"] = dataframe["Z分数"].rolling(window=n).max().shift()

        # 4. 计算自适应布林带
        dataframe["自适应布林上轨"] = dataframe["中轨"] + dataframe["动态倍数"] * dataframe["标准差"]
        dataframe["自适应布林中轨"] = dataframe["中轨"]
        dataframe["自适应布林下轨"] = dataframe["中轨"] - dataframe["动态倍数"] * dataframe["标准差"]

        # 5. 输出计算结果
        if len(dataframe) > 0:
            last = dataframe.iloc[-1]
            logger.info(
                f"{pair} - 计算指标: 收盘价={last['close']:.4f}, 中轨={last['中轨']:.4f}, "
                f"Z分数={last['Z分数']:.4f}, 动态倍数={last['动态倍数']:.4f}, "
                f"上轨={last['自适应布林上轨']:.4f}, 下轨={last['自适应布林下轨']:.4f}"
            )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        logger.info(f"{pair} - 开始计算入场信号")

        # 1. 多头入场条件
        long_cond = (
            (dataframe["close"] > dataframe["自适应布林上轨"]) &  # 价格突破上轨
            (dataframe["close"].shift(1) <= dataframe["自适应布林上轨"].shift(1))  # 前一根未突破
        )

        # 2. 空头入场条件
        short_cond = (
            (dataframe["close"] < dataframe["自适应布林下轨"]) &  # 价格突破下轨
            (dataframe["close"].shift(1) >= dataframe["自适应布林下轨"].shift(1))  # 前一根未突破
        )

        # 3. 设置信号
        dataframe.loc[long_cond, ["enter_long", "enter_tag"]] = (1, "突破自适应布林上轨")
        dataframe.loc[short_cond, ["enter_short", "enter_tag"]] = (1, "突破自适应布林下轨")

        # 4. 输出最新K线的信号值
        if len(dataframe) > 0:
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
        exit_long_cond = (
            (dataframe["close"] < dataframe["中轨"])  # 价格跌破中轨
            & (dataframe["close"].shift(1) >= dataframe["中轨"].shift(1))  # 前一根在中轨上方
        )

        # 2. 空头出场条件
        exit_short_cond = (
            (dataframe["close"] > dataframe["中轨"])  # 价格突破中轨
            & (dataframe["close"].shift(1) <= dataframe["中轨"].shift(1))  # 前一根在中轨下方
        )

        # 3. 设置信号
        dataframe.loc[exit_long_cond, ["exit_long", "exit_tag"]] = (1, "跌破布林中轨")
        dataframe.loc[exit_short_cond, ["exit_short", "exit_tag"]] = (1, "突破布林中轨")

        # 4. 只输出平仓信号
        if len(dataframe) > 0:
            logger.info(
                f"{metadata['pair']} - 计算平仓信号: "
                f"平多={1 if exit_long_cond.iloc[-1] else 0}, "
                f"平空={1 if exit_short_cond.iloc[-1] else 0}"
            )
        return dataframe
