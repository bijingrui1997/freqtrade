from freqtrade.strategy import IStrategy
from pandas import DataFrame
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class MomentumRankStrategy(IStrategy):
    """
    动量排名策略 - 追涨杀跌策略
    
    策略说明:
    本策略通过观察市场动量变化来捕捉趋势机会,同时做多做空以分散风险。
    
    核心逻辑:
    - 每30分钟扫描一次市场,观察过去6小时(12根K线)的涨跌幅变化
    - 选取涨幅最大的2个币种做多,跌幅最大的2个币种做空
    - 每个仓位使用账户10%资金,采用3倍杠杆
    - 固定持仓1天后自动平仓,不设止盈止损
    
    风险控制:
    - 资金分配: 单个币种最多使用10%账户资金
    - 杠杆倍数: 3倍杠杆
    - 持仓时间: 限制为1天,到期强制平仓
    - 开仓条件: 账户资金充足才允许开仓
    
    交易优化:
    - 持仓期间不重复检查信号,避免频繁交易
    - 新信号与当前持仓重叠时,无需重复开平仓
    - 通过减少不必要的开平仓操作来降低手续费支出
    """

    # 策略参数
    LOOKBACK_PERIOD = 12  # 回看周期:6小时(12根30分钟K线)
    TOP_N = 2  # 选择前2名和后2名
    POSITION_SIZE = 0.1  # 每次开仓使用账户10%的资金
    HOLDING_PERIOD = 1  # 持仓1天后自动平仓

    # 基本设置
    timeframe = "30m"  # 使用30分钟K线
    stoploss = -0.99  # 设置一个很大的止损,实际由系统控制风险
    minimal_roi = {
        "0": 100  # 不使用ROI退出
    }
    can_short = True  # 允许做空

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """计算动量指标"""
        # 计算涨跌幅
        dataframe["momentum"] = (
            (dataframe["close"] - dataframe["close"].shift(self.LOOKBACK_PERIOD))
            / dataframe["close"].shift(self.LOOKBACK_PERIOD)
            * 100
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """入场信号"""
        pair = metadata["pair"]

        # 初始化信号列
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        # 确保数据足够
        if len(dataframe) <= self.LOOKBACK_PERIOD:
            return dataframe

        # 对每根K线计算动量排名
        for i in range(self.LOOKBACK_PERIOD, len(dataframe)):
            current_time = dataframe.index[i]

            # 获取所有币对在这个时间点的动量数据
            all_pairs_momentum = {}
            for p in self.dp.current_whitelist():
                pair_dataframe = self.dp.get_pair_dataframe(p, self.timeframe)
                if len(pair_dataframe) > i:  # 确保这个币对在这个时间点有数据
                    momentum = (
                        (
                            pair_dataframe["close"].iloc[i]
                            - pair_dataframe["close"].iloc[i - self.LOOKBACK_PERIOD]
                        )
                        / pair_dataframe["close"].iloc[i - self.LOOKBACK_PERIOD]
                        * 100
                    )
                    all_pairs_momentum[p] = momentum

            if all_pairs_momentum:
                # 按动量排序
                sorted_pairs = sorted(all_pairs_momentum.items(), key=lambda x: x[1], reverse=True)
                top_n = [p[0] for p in sorted_pairs[: self.TOP_N]]  # 最强的N个
                bottom_n = [p[0] for p in sorted_pairs[-self.TOP_N :]]  # 最弱的N个

                # 只在最后一根K线打印日志
                if i == len(dataframe) - 1:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"市场分析报告 - {current_time}")
                    logger.info(
                        f"\n回看周期: 过去{self.LOOKBACK_PERIOD}根{self.timeframe}K线 (约6小时)"
                    )
                    logger.info(f"\n当前监控的{len(sorted_pairs)}个交易对:")
                    for p, m in sorted_pairs:
                        price = self.dp.get_pair_dataframe(p, self.timeframe)["close"].iloc[i]
                        logger.info(f"    {p}: {m:+.2f}% (价格: {price:.4f} USDT)")

                    logger.info(f"\n最强势的{self.TOP_N}个币对(准备做多):")
                    for p in top_n:
                        m = all_pairs_momentum[p]
                        price = self.dp.get_pair_dataframe(p, self.timeframe)["close"].iloc[i]
                        logger.info(f"    {p}: +{m:.2f}% (价格: {price:.4f} USDT)")

                    logger.info(f"\n最弱势的{self.TOP_N}个币对(准备做空):")
                    for p in reversed(bottom_n):
                        m = all_pairs_momentum[p]
                        price = self.dp.get_pair_dataframe(p, self.timeframe)["close"].iloc[i]
                        logger.info(f"    {p}: {m:.2f}% (价格: {price:.4f} USDT)")

                # 设置交易信号
                if pair in top_n:
                    dataframe.loc[dataframe.index[i], "enter_long"] = 1
                    if i == len(dataframe) - 1:  # 只在最后一根K线打印日志
                        price = dataframe["close"].iloc[i]
                        amount = self.wallets.get_total_stake_amount() * self.POSITION_SIZE
                        logger.info(f"\n开仓信号 - 做多 {pair}")
                        logger.info(f"开仓价格: {price:.4f} USDT")
                        logger.info(
                            f"开仓金额: {amount:.2f} USDT (账户总额的{self.POSITION_SIZE*100}%)"
                        )
                        logger.info(f"计划持仓: {self.HOLDING_PERIOD}天")

                if pair in bottom_n:
                    dataframe.loc[dataframe.index[i], "enter_short"] = 1
                    if i == len(dataframe) - 1:  # 只在最后一根K线打印日志
                        price = dataframe["close"].iloc[i]
                        amount = self.wallets.get_total_stake_amount() * self.POSITION_SIZE
                        logger.info(f"\n开仓信号 - 做空 {pair}")
                        logger.info(f"开仓价格: {price:.4f} USDT")
                        logger.info(
                            f"开仓金额: {amount:.2f} USDT (账户总额的{self.POSITION_SIZE*100}%)"
                        )
                        logger.info(f"计划持仓: {self.HOLDING_PERIOD}天")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """出场信号 - 固定持仓时间后平仓"""
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        # 计算持仓时间
        holding_periods = self.HOLDING_PERIOD * 24 * 60 // int(self.timeframe[:-1])  # 转换为K线数量

        # 设置平仓信号
        for i in range(len(dataframe)):
            if i >= holding_periods:
                if dataframe["enter_long"].iloc[i - holding_periods] == 1:
                    dataframe.loc[dataframe.index[i], "exit_long"] = 1
                if dataframe["enter_short"].iloc[i - holding_periods] == 1:
                    dataframe.loc[dataframe.index[i], "exit_short"] = 1

        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """设置杠杆倍数"""
        return 3.0  # 使用3倍杠杆
