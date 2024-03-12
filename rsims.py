import pandas as pd
import numpy as np

def positions_from_no_trade_buffer(current_positions, current_prices, target_weights, cap_equity, trade_buffer):

    num_assets = len(current_positions)
    target_positions = np.zeros(num_assets)
    current_weights = current_positions * current_prices / cap_equity
    
    for j in range(num_assets):
        if np.isnan(target_weights[j]) or target_weights[j] == 0:
            target_positions[j] = 0
        elif current_weights[j] < target_weights[j] - trade_buffer:
            target_positions[j] = (target_weights[j] - trade_buffer) * cap_equity / current_prices[j]
        elif current_weights[j] > target_weights[j] + trade_buffer:
            target_positions[j] = (target_weights[j] + trade_buffer) * cap_equity / current_prices[j]
        else:
            target_positions[j] = current_positions[j]
    return target_positions



def fixed_commission_backtest_with_funding(prices, target_weights, funding_rates, trade_buffer=0.0, initial_cash=10000, commission_pct=0, reinvest=True):

    # Get tickers for later
    tickers = prices.columns[1:]

    # Initial state
    num_assets = prices.shape[1] - 1  # -1 for date column
    current_positions = np.zeros(num_assets)
    previous_target_weights = np.zeros(num_assets)
    previous_prices = np.full(num_assets, 0)
    rows_list = []
    cash = initial_cash
    total_eq = np.array(0)

    # Backtest loop
    for i in range(len(prices)):
        # fetch data
        current_date = prices.iloc[i, 0]
        current_prices = prices.iloc[i, 1:]
        current_target_weights = target_weights.iloc[i, 1:]
        current_funding_rates = funding_rates.iloc[i, 1:]

        # Accrue funding on current positions
        funding = current_positions * current_prices.values * current_funding_rates.values
        funding = np.where(funding == np.nan, 0, funding)
        # PnL for the period: price change + funding
        period_pnl = current_positions * (current_prices.values - previous_prices)
        period_pnl = np.where(period_pnl == np.nan, 0, period_pnl) + funding
        # Update cash balance - includes adding back yesterday's margin and deducting today's margin
        cash += np.sum(period_pnl)
        # Update equity
        cap_equity = min(initial_cash, cash) if not reinvest else cash
        # Update positions based on no-trade buffer
        target_positions = positions_from_no_trade_buffer(current_positions, current_prices, current_target_weights, cap_equity, trade_buffer)
        # Calculate position deltas, trade values, and commissions
        trades = target_positions - current_positions

        trade_value = trades * current_prices.dropna().values
        commissions = np.abs(trade_value) * commission_pct
        # After each iteration, set positions to current positions
        current_positions = target_positions
        position_value = current_positions * current_prices.dropna().values

        # Update cash with impact from commissions
        cash -= np.sum(commissions)
        total_eq = np.append(total_eq, cash)

        row_dict = {
            "Date": [current_date] * (num_assets),
            "Close": current_prices.dropna().values,
            "Position": current_positions,
            "Value": position_value,
            "Funding": funding,
            "PeriodPnL": period_pnl,
            "Trades": trades,  
            "TradeValue": trade_value,
            "Commission": commissions,
        }
        rows_list.append(row_dict)
        previous_prices = current_prices
    
    # Combine list of dictionaries into a DataFrame
    result_df = pd.DataFrame(rows_list)
    result_df = result_df.set_index(result_df.Date.apply(lambda x:x[0])).drop(['Date'],axis=1)
    total_eq = pd.Series(total_eq[1:],index=result_df.index)
    sharpe = total_eq.pct_change().mean()/total_eq.pct_change().std()*16
    return result_df, total_eq, sharpe


def split_column_lists(df, tickers):
    result_dfs = {}
    for col in df.columns:
        # Apply function to split lists into separate columns
        result_df = df[col].apply(pd.Series)
        # Rename columns with tickers
        result_df.columns = tickers
        # Store the resulting DataFrame
        result_dfs[col] = result_df
    return result_dfs
