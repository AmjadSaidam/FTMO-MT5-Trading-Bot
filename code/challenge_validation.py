import pandas as pd

class PropFirmValidation():
    """
    prop firm constraint checking functions
    """
    def __init__(self, 
                 strategy_returns: pd.Series, 
                 strategy_equity: pd.Series, 
                 starting_balance: float = 1e4):
        self.rt = strategy_returns
        self.eq = strategy_equity
        self.acc_bal = starting_balance

        # default challenge setting 
        self.eod_trailing = False 
        self.percent_profit_target = 0.1 
        self.percent_total_loss = 0.1
        self.max_daily_loss = 0.03
        self.best_day_ceiling = 1.0
        
        # logs 
        self.strategy_score = None 

        self.frac_max_profit_realised = 0
        self.frac_max_loss_unrealised = 1
        self.frac_max_daily_loss_unrealised = 1
        self.frac_pnl_of_max_pnl_share_realised = 0

    def strategy_challenge_summary(self): 
        """
        simulate passing prof-firm trading challenge 
        """
        challenge_stats = {}

        passed_tot_loss_pnl = self.max_absolute_loss_or_profit()
        challenge_stats['passed_max_total_loss'] = passed_tot_loss_pnl[0]
        challenge_stats['passed_profit_target'] = passed_tot_loss_pnl[1]
        challenge_stats['passed_max_daily_loss'] = self.max_daily_loss_()
        challenge_stats['passed_best_day_pnl'] = self.best_day_pnl_()

        self.strategy_score = min(self.frac_max_profit_realised, 2.0) + self.frac_max_loss_unrealised + self.frac_max_daily_loss_unrealised + self.frac_pnl_of_max_pnl_share_realised 

        return challenge_stats

    # --- helpers ---
    def max_absolute_loss_or_profit(self) -> tuple[bool, bool]:
        """
        checks is maximum absolute (total) account loss or target profit are passed, returs False otherwise 
        """
        passed_max_tot_loss = True
        passed_profit_target = False

        # logs 
        max_ror = (self.eq.max() - self.acc_bal) / self.acc_bal
        # max_dd = (self.acc_bal - self.eq.min()) / self.eq.min()

        # fraction of profit target realised
        frac_pnl_target = max_ror / self.percent_profit_target
        self.frac_max_profit_realised += frac_pnl_target # must be clipped to ensure not over under pernalisig risk metric in strategy score
        self.frac_max_loss_unrealised += frac_pnl_target
        
        equity_run_up = self.acc_bal

        eod_loss = self.acc_bal - self.acc_bal * self.percent_total_loss

        profit_target = self.acc_bal + self.acc_bal * self.percent_profit_target

        # event based (returns if failed)
        for eq in self.eq.to_numpy():
            # equity run-up/peak
            if self.eod_trailing:
                if eq > equity_run_up:
                    equity_run_up = eq
                    eod_loss = equity_run_up - equity_run_up * self.percent_total_loss
            # equity draw-down 
            if eq < eod_loss: 
                passed_max_tot_loss = False
                return (passed_max_tot_loss, passed_profit_target)
            
            # check if profit target has been hit 
            if eq > profit_target:
                passed_profit_target = True 
                return (passed_max_tot_loss, passed_profit_target)
            
        # default case 
        return (passed_max_tot_loss, passed_profit_target)

    def max_daily_loss_(self):
        """
        maximum daily loss
        """
        daily_sum = self.rt.resample('D').sum()

        if (daily_sum < -self.max_daily_loss).any():
            return False 
        
        frac_max_daily_loss = daily_sum.min() / self.max_daily_loss
        self.frac_max_daily_loss_unrealised += frac_max_daily_loss

        return True
    
    def best_day_pnl_(self):
        """
        best day positive return must contribute at most best_day_celining of total positive profits 

        this avoids passing a challenge with exactlye one signle big trade
        """
        total_positive = self.rt[self.rt > 0].sum()

        if total_positive <= 0: # no winning days - ratio is undefined, not a violation
            return True

        frac_best_pnl = self.rt.max() / total_positive
        invalidate_frac_max_dailt_pnl = frac_best_pnl < self.best_day_ceiling

        pnl_share_from_target = (self.best_day_ceiling - frac_best_pnl) / self.best_day_ceiling
        self.frac_pnl_of_max_pnl_share_realised += pnl_share_from_target

        return invalidate_frac_max_dailt_pnl