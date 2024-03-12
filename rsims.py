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

    return target_positions



def fixed_commission_backtest_with_funding(prices, target_weights, funding_rates, trade_buffer=0.0, initial_cash=10000, commission_pct=0, reinvest=True):
    if trade_buffer < 0:
        raise ValueError("trade_buffer must be greater than or equal to zero")

    misaligned_timestamps_prices = prices.iloc[:, 0] != target_weights.iloc[:, 0]
    misaligned_timestamps_funding = prices.iloc[:, 0] != funding_rates.iloc[:, 0]

    if misaligned_timestamps_prices.any():
        raise ValueError(f"Prices timestamps misaligned with target weights timestamps at prices indexes {misaligned_timestamps_prices}")

    if misaligned_timestamps_funding.any():
        raise ValueError(f"Prices timestamps misaligned with funding rates timestamps at prices indexes {misaligned_timestamps_funding}")

    if not prices.shape == target_weights.shape:
        raise ValueError("Prices and weights matrices must have the same dimensions")

    if not prices.shape == funding_rates.shape:
        raise ValueError("Prices and funding matrices must have the same dimensions")

    # Check for NA in weights and funding matrices
    if target_weights.isna().any().any():
        print("NA present in target weights: consider replacing these values before continuing")

    if funding_rates.isna().any().any():
        print("NA present in funding rates: consider replacing these values before continuing")

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

        # Post-trade cash: cash freed up from closing positions, cash used as margin, commissions
        target_position_value = target_positions * current_prices.dropna().values
        post_trade_cash = cash - np.sum(commissions) 

        current_positions = target_positions
        position_value = current_positions * current_prices.dropna().values

        # Update cash
        cash = post_trade_cash
        total_eq = np.append(total_eq, post_trade_cash)

        row_dict = {

            "Date": [current_date] * (num_assets),
            "Close": current_prices.dropna().values,
            "Position": current_positions,
            "Value": position_value,
            "Funding": funding,
            "PeriodPnL": period_pnl,
            "Trades": trades,  # Minus because we keep the sign of the original position in liq_contracts
            "TradeValue": trade_value,
            "Commission": commissions,
        }



        rows_list.append(row_dict)

        previous_target_weights = current_target_weights
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
