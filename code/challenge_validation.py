import numpy as np
import pandas as pd

class PropFirmValidation():
    """
    prop firm constraint checking functions

    strategy_score is higher for a better strategy:
        score = profit_component + total_loss_headroom + daily_loss_headroom + best_day_headroom
        each component is clipped to a bounded range, and any rule breach subtracts
        failure_penalty so a failing strategy always ranks below a passing one.
    """
    def __init__(self,
                 strategy_returns: pd.Series,
                 strategy_equity: pd.Series,
                 starting_balance: float = 1e4):
        if not isinstance(strategy_equity.index, pd.DatetimeIndex):
            raise TypeError("strategy_equity must have a DatetimeIndex (needed for daily resampling)")

        self.rt = strategy_returns
        self.eq = strategy_equity
        self.acc_bal = starting_balance

        # default challenge setting
        self.eod_trailing = True
        self.percent_profit_target = 0.1
        self.percent_total_loss = 0.1
        self.check_daily_loss = True # set True to enforce and score max_daily_loss
        self.max_daily_loss = 0.03
        self.best_day_ceiling = 0.5 # np.inf -> best trade return cant be infinity

        # scoring
        self.max_profit_component = 2.0 # cap on profit-target multiples rewarded
        self.failure_penalty = 10.0     # > max spread of the bounded components (sum lies in [-4, 5])

        # logs
        self.strategy_score = None
        self._reset_score_components()

    def _reset_score_components(self):
        self.frac_max_profit_realised = 0.0           # max run-up / profit target, clipped to [-1, max_profit_component]
        self.frac_max_loss_unrealised = 0.0           # unused share of total loss allowance, clipped to [-1, 1]
        self.frac_max_daily_loss_unrealised = 0.0     # unused share of daily loss allowance, clipped to [-1, 1]
        self.frac_pnl_of_max_pnl_share_realised = 0.0 # headroom below best-day ceiling, clipped to [-1, 1]

    def strategy_challenge_summary(self):
        """simulate passing prof-firm trading challenge"""
        self._reset_score_components()
        challenge_stats = {}

        passed_tot_loss_pnl = self.max_absolute_loss_or_profit()
        challenge_stats['passed_max_total_loss'] = passed_tot_loss_pnl[0]
        challenge_stats['passed_profit_target'] = passed_tot_loss_pnl[1]
        challenge_stats['passed_max_daily_loss'] = self.max_daily_loss_() if self.check_daily_loss else True
        challenge_stats['passed_best_day_pnl'] = self.best_day_pnl_()

        score = (self.frac_max_profit_realised
                 + self.frac_max_loss_unrealised
                 + self.frac_max_daily_loss_unrealised
                 + self.frac_pnl_of_max_pnl_share_realised)

        hard_rules = ('passed_max_total_loss', 'passed_max_daily_loss', 'passed_best_day_pnl')
        if not all(challenge_stats[rule] for rule in hard_rules):
            score -= self.failure_penalty

        self.strategy_score = score

        return challenge_stats

    # --- helpers ---
    def _daily_close(self) -> pd.Series:
        """end-of-day equity (trading days only)"""
        return self.eq.resample('D').last().dropna()

    def _total_loss_floor(self) -> np.ndarray:
        """per-bar equity floor for the max total loss rule (static, or trailing end-of-day peak)"""
        static_floor = self.acc_bal * (1 - self.percent_total_loss)
        if not self.eod_trailing:
            return np.full(len(self.eq), static_floor)

        # peak of previous end-of-day equities, applied from the next day onwards
        prev_peak = self._daily_close().cummax().shift(1).clip(lower = self.acc_bal).fillna(self.acc_bal)
        floor_daily = prev_peak * (1 - self.percent_total_loss)

        return floor_daily.reindex(self.eq.index.floor('D')).fillna(static_floor).to_numpy()

    def max_absolute_loss_or_profit(self) -> tuple[bool, bool]:
        """
        checks whether the max total loss floor or the profit target is hit first, returns
        (passed_max_total_loss, passed_profit_target)
        """
        eq = self.eq.to_numpy(dtype = float)
        floor = self._total_loss_floor()
        profit_target = self.acc_bal * (1 + self.percent_profit_target)

        # score: fraction of profit target realised (max run-up)
        max_ror = (eq.max() - self.acc_bal) / self.acc_bal
        self.frac_max_profit_realised = float(np.clip(max_ror / self.percent_profit_target,
                                                      -1.0, self.max_profit_component))

        # score: worst unused share of the loss allowance (1 = never drew down, 0 = touched floor, < 0 = breached)
        allowance = floor / (1 - self.percent_total_loss) * self.percent_total_loss
        self.frac_max_loss_unrealised = float(np.clip(((eq - floor) / allowance).min(), -1.0, 1.0))

        # event based: whichever rule is hit first ends the challenge
        breach = np.flatnonzero(eq < floor)
        target = np.flatnonzero(eq >= profit_target)
        first_breach = breach[0] if breach.size else np.inf
        first_target = target[0] if target.size else np.inf

        if first_breach < first_target:
            return (False, False)
        
        return (True, bool(first_target < np.inf))

    def max_daily_loss_(self) -> bool:
        """maximum daily loss: intraday equity low vs start-of-day (previous close) equity"""
        day_min = self.eq.resample('D').min().dropna()
        day_start = self._daily_close().shift(1).fillna(self.acc_bal)
        daily_loss = (day_min - day_start) / day_start # <= 0 on losing days

        worst = min(daily_loss.min(), 0.0)
        self.frac_max_daily_loss_unrealised = float(np.clip(1 + worst / self.max_daily_loss, -1.0, 1.0))

        return bool(worst >= -self.max_daily_loss)

    def best_day_pnl_(self) -> bool:
        """
        best day profit must contribute at most best_day_ceiling of total positive daily profits

        this avoids passing a challenge with exactly one single big trade
        """
        daily_close = self._daily_close()
        daily_pnl = daily_close.diff()
        daily_pnl.iloc[0] = daily_close.iloc[0] - self.acc_bal

        total_positive = daily_pnl[daily_pnl > 0].sum()
        if total_positive <= 0: # no winning days - ratio is undefined, not a violation (profit score handles it)
            return True

        frac_best_pnl = daily_pnl.max() / total_positive
        self.frac_pnl_of_max_pnl_share_realised = float(np.clip(
            (self.best_day_ceiling - frac_best_pnl) / self.best_day_ceiling, -1.0, 1.0))

        return bool(frac_best_pnl <= self.best_day_ceiling)
