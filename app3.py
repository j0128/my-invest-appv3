import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 0. 全局設定 ---
st.set_page_config(page_title="Alpha 12.3: 策略實驗室", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #444; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
    .stTabs [data-baseweb="tab-list"] {gap: 5px;}
    .stTabs [data-baseweb="tab"] {height: 50px; background-color: #1E1E1E; border-radius: 5px 5px 0 0; color: white;}
    .stTabs [aria-selected="true"] {background-color: #00BFFF; color: black;}
</style>
""", unsafe_allow_html=True)

# --- 1. 核心數據引擎 ---
@st.cache_data(ttl=1800)
def fetch_market_data(tickers):
    benchmarks = ['SPY', 'QQQ', 'QLD', 'TQQQ', '^VIX', '^TNX', '^IRX', 'HYG', 'GC=F', 'HG=F', 'DX-Y.NYB'] 
    all_tickers = list(set(tickers + benchmarks))
    data = {col: {} for col in ['Close', 'Open', 'High', 'Low', 'Volume']}
    
    for i, t in enumerate(all_tickers):
        try:
            df = yf.Ticker(t).history(period="2y", auto_adjust=True)
            if df.empty: continue
            data['Close'][t] = df['Close']
            data['Open'][t] = df['Open']
            data['High'][t] = df['High']
            data['Low'][t] = df['Low']
            data['Volume'][t] = df['Volume']
        except: continue
    
    try:
        return (pd.DataFrame(data['Close']).ffill(), pd.DataFrame(data['High']).ffill(), 
                pd.DataFrame(data['Low']).ffill(), pd.DataFrame(data['Volume']).ffill())
    except: return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

@st.cache_data(ttl=3600*12)
def fetch_fred_macro(api_key):
    if not api_key: return None
    try:
        fred = Fred(api_key=api_key)
        walcl = fred.get_series('WALCL', observation_start='2024-01-01')
        tga = fred.get_series('WTREGEN', observation_start='2024-01-01')
        rrp = fred.get_series('RRPONTSYD', observation_start='2024-01-01')
        fed_rate = fred.get_series('FEDFUNDS', observation_start='2023-01-01')
        df = pd.DataFrame({'WALCL': walcl, 'TGA': tga, 'RRP': rrp}).ffill().dropna()
        df['Net_Liquidity'] = (df['WALCL'] - df['TGA'] - df['RRP']) / 1000 
        df_rate = pd.DataFrame({'Fed_Rate': fed_rate}).resample('D').ffill()
        return df, df_rate
    except: return None, None

@st.cache_data(ttl=3600*24)
def get_advanced_info(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        q_type = info.get('quoteType', '').upper()
        is_etf = 'ETF' in q_type or 'MUTUALFUND' in q_type
        peg = info.get('pegRatio')
        fwd_pe = info.get('forwardPE')
        earn_growth = info.get('earningsGrowth')
        if peg is None and fwd_pe is not None and earn_growth is not None and earn_growth > 0:
            peg = fwd_pe / (earn_growth * 100)
        rev_g = info.get('revenueGrowth')
        pm = info.get('profitMargins')
        r40 = (rev_g + pm) * 100 if (rev_g is not None and pm is not None) else None
        return {
            'Type': 'ETF' if is_etf else 'Stock',
            'Target_Mean': info.get('targetMeanPrice'), 
            'Forward_PE': fwd_pe,
            'PEG': peg,
            'Inst_Held': info.get('heldPercentInstitutions'),
            'Insider_Held': info.get('heldPercentInsiders'),
            'Short_Ratio': info.get('shortRatio'),
            'Current_Ratio': info.get('currentRatio'),
            'Debt_Equity': info.get('debtToEquity'),
            'ROE': info.get('returnOnEquity'),
            'Profit_Margin': pm,
            'Rule_40': r40
        }
    except: return {'Type': 'Unknown'}

# --- 2. 戰略運算 ---

def train_rf_model(df_close, ticker, days_forecast=22):
    try:
        if ticker not in df_close.columns: return None
        df = pd.DataFrame(index=df_close.index)
        df['Close'] = df_close[ticker]
        df['Ret'] = df['Close'].pct_change()
        df['Vol'] = df['Ret'].rolling(20).std()
        df['SMA'] = df['Close'].rolling(20).mean()
        if '^VIX' in df_close.columns: df['VIX'] = df_close['^VIX']
        if '^TNX' in df_close.columns: df['TNX'] = df_close['^TNX']
        df['Target'] = df['Close'].shift(-days_forecast)
        df = df.dropna()
        if len(df) < 60: return None
        X = df.drop(columns=['Target', 'Close'])
        y = df['Target']
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X, y)
        return model.predict(X.iloc[[-1]])[0]
    except: return None

def calc_targets_composite(ticker, df_close, df_high, df_low, f_data, days_forecast=22):
    if ticker not in df_close.columns: return None
    c = df_close[ticker]; h = df_high[ticker]; l = df_low[ticker]
    if len(c) < 100: return None
    try:
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        t_atr = c.iloc[-1] + (atr * np.sqrt(days_forecast))
    except: t_atr = None
    try:
        mu = c.pct_change().mean()
        t_mc = c.iloc[-1] * ((1 + mu)**days_forecast)
    except: t_mc = None
    try:
        recent = c.iloc[-60:]
        t_fib = recent.max() + (recent.max() - recent.min()) * 0.618 
    except: t_fib = None
    t_fund = f_data.get('Target_Mean')
    t_rf = train_rf_model(df_close, ticker, days_forecast)
    targets = [t for t in [t_atr, t_mc, t_fib, t_fund, t_rf] if t is not None and not pd.isna(t)]
    t_avg = sum(targets) / len(targets) if targets else None
    return {"ATR": t_atr, "MC": t_mc, "Fib": t_fib, "Fund": t_fund, "RF": t_rf, "Avg": t_avg}

def run_backtest_lab(ticker, df_close, df_high, df_low, days_ago=22):
    if ticker not in df_close.columns or len(df_close) < 250: return None
    idx_past = len(df_close) - days_ago - 1
    p_now = df_close[ticker].iloc[-1]
    df_past = df_close.iloc[:idx_past+1]
    past_rf = train_rf_model(df_past, ticker, days_ago)
    c_slice = df_close[ticker].iloc[:idx_past+1]
    h_slice = df_high[ticker].iloc[:idx_past+1]
    l_slice = df_low[ticker].iloc[:idx_past+1]
    tr = pd.concat([h_slice-l_slice], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    past_atr = c_slice.iloc[-1] + (atr * np.sqrt(days_ago))
    past_mc = c_slice.iloc[-1] * ((1 + c_slice.pct_change().mean())**days_ago)
    valid_past = [x for x in [past_rf, past_atr, past_mc] if x is not None]
    if not valid_past: return None
    past_avg = sum(valid_past) / len(valid_past)
    err = (past_avg - p_now) / p_now
    return {"Past_Pred": past_avg, "Error": err}

def calc_mvrv_z(series):
    if len(series) < 200: return None
    sma200 = series.rolling(200).mean()
    std200 = series.rolling(200).std()
    return (series - sma200) / std200

def calc_tech_indicators(series, vol_series):
    if len(series) < 60: return None, None, None
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    ma20 = series.rolling(20).mean()
    slope = (ma20.iloc[-1] - ma20.iloc[-5]) / ma20.iloc[-5]
    vol_ma = vol_series.rolling(20).mean().iloc[-1]
    vol_ratio = vol_series.iloc[-1] / vol_ma if vol_ma > 0 else 1.0
    return rsi, slope, vol_ratio

def calc_six_dim_state(series):
    if len(series) < 22: return "N/A"
    p = series.iloc[-1]
    ma20 = series.rolling(20).mean().iloc[-1]
    std20 = series.rolling(20).std().iloc[-1]
    up = ma20 + 2 * std20
    lw = ma20 - 2 * std20
    if p > up * 1.05: return "H3 極限噴出"
    if p > up: return "H2 情緒過熱"
    if p > ma20: return "H1 多頭回歸"
    if p < lw * 0.95: return "L3 恐慌崩盤"
    if p < lw: return "L2 超賣區"
    return "L1 震盪整理"

def get_cfo_directive_v3(p_now, six_state, trend_status, range_high, range_low, mvrv_z, rsi, slope, vol_ratio):
    if "L" in six_state and "空頭" in trend_status: return "⬛ 趨勢損毀 (清倉)"
    if ("H3" in six_state) or (rsi is not None and rsi > 80): return "🟥 極限噴出 (賣1/2)"
    if range_high > 0 and p_now >= range_high: return "🟥 達預測高點 (賣1/2)"
    if "H2" in six_state: return "🟧 過熱減碼 (賣1/3)"
    
    buy_signals = []
    if (mvrv_z is not None and mvrv_z < -0.5) or (range_low > 0 and p_now < range_low): buy_signals.append("🔵 價值買點")
    if "L2" in six_state: buy_signals.append("💎 抄底機會 (30%)")
    if "多頭" in trend_status and ("H1" in six_state or "L1" in six_state):
        if slope is not None and slope > 0.01 and vol_ratio > 1.5: buy_signals.append("🔥 加速進攻 (80%)")
        elif slope is not None and slope > 0: buy_signals.append("🟢 多頭確立 (50%)")
        else: buy_signals.append("🟢 轉強試單 (20%)")
    return " | ".join(buy_signals) if buy_signals else "⬜ 觀望/持有"

def analyze_trend_multi(series):
    if series is None or len(series) < 126: return {}
    y = series.iloc[-126:].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    p_now = series.iloc[-1]
    sma200 = series.rolling(200).mean().iloc[-1]
    status = "🔥 多頭" if p_now > sma200 else "🛑 空頭"
    if p_now < sma200 and p_now > sma200 * 0.9: status = "📉 弱勢"
    return {"p_1m": model.predict([[len(y)+22]])[0].item(), "p_now": p_now, "status": status}

def calc_dynamic_kelly(series, lookback=63):
    try:
        if len(series) < lookback: return 0.0
        recent = series.iloc[-lookback:]
        rets = recent.pct_change().dropna()
        if len(rets) < 10: return 0.0
        wins = rets[rets > 0]; losses = rets[rets < 0]
        if len(losses) == 0: return 1.0 
        if len(wins) == 0: return 0.0
        win_rate = len(wins) / len(rets)
        avg_win = wins.mean(); avg_loss = abs(losses.mean())
        if avg_loss == 0: return 1.0
        wl_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / wl_ratio)
        return max(0.0, min(1.0, kelly * 0.5))
    except: return 0.0

def calc_obv(close, volume):
    if volume is None: return None
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

def compare_with_leverage(ticker, df_close):
    if ticker not in df_close.columns: return None
    benchs = ['QQQ', 'QLD', 'TQQQ']
    valid_benchs = [b for b in benchs if b in df_close.columns]
    if not valid_benchs: return None
    lookback = 252
    if len(df_close) < lookback: lookback = len(df_close)
    df_compare = df_close[[ticker] + valid_benchs].iloc[-lookback:].copy()
    df_norm = df_compare / df_compare.iloc[0] * 100
    ret_ticker = df_norm[ticker].iloc[-1] - 100
    ret_tqqq = df_norm['TQQQ'].iloc[-1] - 100 if 'TQQQ' in df_norm else 0
    status = "👑 跑贏 TQQQ" if ret_ticker > ret_tqqq else "💀 輸給 TQQQ"
    return df_norm, status, ret_ticker, ret_tqqq

# [NEW] 策略實驗室回測引擎
def run_strategy_backtest(df_in, frequency_days=1):
    """
    回測邏輯：
    1. 起始資金 10000，每月1號 +10000。
    2. DCA 組：錢進來直接買。
    3. 策略組：根據頻率檢測訊號 (CFO V3 邏輯)，決定買賣比例。
       - 買進：根據 Cash 的 % (20%, 50%, 80%)
       - 賣出：根據 持倉 的 % (33%, 50%, 100%)
    """
    df = df_in.copy()
    # 預先計算指標以加速
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['STD20'] = df['Close'].rolling(20).std()
    df['Upper'] = df['SMA20'] + 2 * df['STD20']
    df['Lower'] = df['SMA20'] - 2 * df['STD20']
    
    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    df['RSI'] = 100 - (100 / (1 + up.ewm(com=13).mean() / down.ewm(com=13).mean()))
    
    # Slope
    df['Slope'] = (df['SMA20'] - df['SMA20'].shift(5)) / df['SMA20'].shift(5)

    # 初始化
    cash_dca = 0; shares_dca = 0; invested_dca = 0
    cash_strat = 0; shares_strat = 0; invested_strat = 0
    
    history = []
    last_month = -1
    
    # 僅跑最後 300 天
    if len(df) > 300: df = df.iloc[-300:]
    
    for i in range(len(df)):
        date = df.index[i]
        price = df['Close'].iloc[i]
        
        # 1. 每月發薪水 (每月第1天或最接近的一天)
        if date.month != last_month:
            # Inject Capital
            cash_dca += 10000
            invested_dca += 10000
            
            cash_strat += 10000
            invested_strat += 10000
            
            # DCA 策略：有錢就買
            can_buy = cash_dca // price
            if can_buy > 0:
                shares_dca += can_buy
                cash_dca -= can_buy * price
            
            last_month = date.month
        
        # 2. 策略組交易 (按頻率)
        if i % frequency_days == 0 and i > 20: # 確保有均線數據
            # 判斷訊號 (簡化版 V3 邏輯，避免 look-ahead)
            p = price; ma20 = df['SMA20'].iloc[i]; upper = df['Upper'].iloc[i]; lower = df['Lower'].iloc[i]
            ma200 = df['SMA200'].iloc[i]; rsi = df['RSI'].iloc[i]; slope = df['Slope'].iloc[i]
            
            # 賣訊
            sell_pct = 0
            if p < ma20 and ma200 > 0 and p < ma200: sell_pct = 1.0 # 趨勢損毀
            elif p > upper * 1.05 or rsi > 80: sell_pct = 0.5 # 極限噴出
            elif p > upper: sell_pct = 0.33 # 過熱
            
            if sell_pct > 0 and shares_strat > 0:
                sell_amt = int(shares_strat * sell_pct)
                if sell_amt > 0:
                    shares_strat -= sell_amt
                    cash_strat += sell_amt * price
            
            # 買訊
            buy_pct_cash = 0
            # 抄底
            if p < lower: buy_pct_cash = 0.3
            # 順勢
            elif p > ma20 and p > ma200: # 簡化：多頭
                if slope > 0.01: buy_pct_cash = 0.8 # 加速
                elif slope > 0: buy_pct_cash = 0.5 # 確立
                else: buy_pct_cash = 0.2 # 試單
            
            if buy_pct_cash > 0 and cash_strat > 0:
                amount_to_spend = cash_strat * buy_pct_cash
                can_buy = amount_to_spend // price
                if can_buy > 0:
                    shares_strat += can_buy
                    cash_strat -= can_buy * price

        # 記錄淨值
        val_dca = cash_dca + shares_dca * price
        val_strat = cash_strat + shares_strat * price
        history.append({
            "Date": date,
            "DCA_Value": val_dca,
            "Strat_Value": val_strat,
            "Invested": invested_strat
        })
        
    res_df = pd.DataFrame(history).set_index("Date")
    
    # 計算最終指標
    final_dca = res_df['DCA_Value'].iloc[-1]
    final_strat = res_df['Strat_Value'].iloc[-1]
    invested = res_df['Invested'].iloc[-1]
    
    roi_dca = (final_dca - invested) / invested
    roi_strat = (final_strat - invested) / invested
    
    return res_df, roi_dca, roi_strat, invested, final_dca, final_strat

# --- 3. 財務計算 ---
def run_traffic_light(series):
    sma200 = series.rolling(200).mean()
    df = pd.DataFrame({'Close': series, 'SMA200': sma200})
    df['Signal'] = np.where(df['Close'] > df['SMA200'], 1, 0)
    df['Strategy'] = (1 + df['Close'].pct_change() * df['Signal'].shift(1)).cumprod()
    df['BuyHold'] = (1 + df['Close'].pct_change()).cumprod()
    return df['Strategy'], df['BuyHold']

def calc_coast_fire(age, r_age, net, save, rate, inf):
    years = r_age - age
    real = (1 + rate/100)/(1 + inf/100) - 1
    data = []
    bal = net
    for y in range(years+1):
        data.append({"Age": age+y, "Balance": bal})
        bal = bal*(1+real) + save*12
    return bal, pd.DataFrame(data)

def calc_mortgage(amt, yrs, rate):
    r = rate/100/12; m = yrs*12
    pmt = amt * (r * (1 + r)**m) / ((1 + r)**m - 1) if r > 0 else amt/m
    return pmt, pmt*m - amt

def parse_input(text):
    port = {}
    for line in text.strip().split('\n'):
        if ',' in line:
            parts = line.split(',')
            try: port[parts[0].strip().upper()] = float(parts[1].strip())
            except: port[parts[0].strip().upper()] = 0.0
    return port

# --- MAIN APP ---
def main():
    with st.sidebar:
        st.header("⚙️ 設定")
        fred_key = st.secrets.get("FRED_API_KEY", st.text_input("FRED API Key", type="password"))
        default_input = """BTC-USD, 10000\nAMD, 10000\nNVDA, 10000\nTLT, 5000\nURA, 5000"""
        user_input = st.text_area("持倉清單", default_input, height=150)
        portfolio_dict = parse_input(user_input)
        tickers_list = list(portfolio_dict.keys())
        total_value = sum(portfolio_dict.values())
        st.metric("總資產 (Est.)", f"${total_value:,.0f}")
        if st.button("🚀 啟動實驗室", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False): return

    with st.spinner("🦅 Alpha 12.3 正在進行時空回測..."):
        df_close, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_macro, df_fed = fetch_fred_macro(fred_key)
        adv_data = {t: get_advanced_info(t) for t in tickers_list}

    if df_close.empty: st.error("❌ 無數據"); st.stop()

    t1, t2, t3, t4, t5, t6, t7 = st.tabs(["🦅 戰略戰情", "🐋 深度籌碼", "🔍 個股體檢", "🚦 策略回測", "💰 CFO 財報", "🏠 房貸目標", "📈 策略實驗室"])

    # === TAB 1: 戰略 ===
    with t1:
        st.subheader("1. 宏觀與總表")
        liq = df_macro['Net_Liquidity'].iloc[-1] if df_macro is not None else 0
        vix = df_close['^VIX'].iloc[-1] if '^VIX' in df_close.columns else 0
        tnx = df_close['^TNX'].iloc[-1] if '^TNX' in df_close.columns else 0
        try: cg = (df_close['HG=F'].iloc[-1]/df_close['GC=F'].iloc[-1])*1000
        except: cg = 0
        
        if df_fed is not None and not df_fed.empty:
            curr_rate = df_fed['Fed_Rate'].iloc[-1]
            past_rate = df_fed['Fed_Rate'].iloc[-90]
            if curr_rate > past_rate + 0.1: rate_dir = "🔺 升息"
            elif curr_rate < past_rate - 0.1: rate_dir = "🔻 降息"
            else: rate_dir = "➡️ 維持"
        else:
            curr_rate = df_close['^IRX'].iloc[-1] if '^IRX' in df_close.columns else 0
            rate_dir = "短債預期"

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("💧 淨流動性", f"${liq:.2f}T")
        c2.metric("🌪️ VIX", f"{vix:.2f}", delta_color="inverse")
        c3.metric("⚖️ 10年殖利率", f"{tnx:.2f}%")
        c4.metric("🏭 銅金比", f"{cg:.2f}")
        c5.metric("🏦 基準利率", f"{curr_rate:.2f}%", rate_dir)

        if df_macro is not None: st.plotly_chart(px.line(df_macro, y='Net_Liquidity', title='聯準會流動性趨勢', height=250), use_container_width=True)

        st.markdown("#### 📊 CFO 戰略指令總表")
        summary = []
        for t in tickers_list:
            if t not in df_close.columns: continue
            
            trend = analyze_trend_multi(df_close[t])
            targets = calc_targets_composite(t, df_close, df_high, df_low, adv_data.get(t,{}), 22)
            bt = run_backtest_lab(t, df_close, df_high, df_low, 22)
            six_state = calc_six_dim_state(df_close[t])
            d_kelly = calc_dynamic_kelly(df_close[t], 63)
            mvrv_s = calc_mvrv_z(df_close[t])
            mvrv_z = mvrv_s.iloc[-1] if mvrv_s is not None else 0
            rsi, slope, vol_r = calc_tech_indicators(df_close[t], df_vol[t])
            
            vol_daily = df_close[t].pct_change().std()
            price_sigma = df_close[t].iloc[-1] * vol_daily * np.sqrt(22)
            tgt_val = targets['Avg'] if targets and targets['Avg'] else 0
            range_low = 0; range_high = 0; range_str = "-"
            if tgt_val > 0:
                range_low = tgt_val - 2 * price_sigma
                range_high = tgt_val + 2 * price_sigma
                range_str = f"${range_low:.0f} ~ ${range_high:.0f}"
            
            cfo_act = get_cfo_directive_v3(trend['p_now'], six_state, trend['status'], range_high, range_low, mvrv_z, rsi, slope, vol_r)
            
            kelly_s = f"{d_kelly*100:.1f}%"
            if d_kelly == 0: kelly_s = "🛑 0%"
            elif d_kelly > 0.5: kelly_s = f"🔥 {d_kelly*100:.0f}%"
            
            summary.append({
                "代號": t, "現價": f"${trend['p_now']:.2f}", 
                "CFO 指令": cfo_act, "動態 Kelly": kelly_s,
                "預測值 (1M)": f"${tgt_val:.2f}" if tgt_val > 0 else "-",
                "95% 區間": range_str, "狀態": six_state,
                "MVRV (Z)": f"{mvrv_z:.2f}", "回測 Bias": f"{bt['Error']:.1%}" if bt else "-"
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True)

    # === TAB 2~6 (保留) ===
    with t2:
        st.subheader("🐋 籌碼與內部人")
        chip_data = []
        for t in tickers_list:
            if t not in df_close.columns: continue
            info = adv_data.get(t, {})
            inst = info.get('Inst_Held'); insider = info.get('Insider_Held'); short = info.get('Short_Ratio')
            chip_data.append({"代號": t, "機構持股": f"{inst*100:.1f}%" if inst is not None else "-", "內部人持股": f"{insider*100:.1f}%" if insider is not None else "-", "空單比例": f"{short:.2f}" if short is not None else "-"})
        st.dataframe(pd.DataFrame(chip_data), use_container_width=True)
    with t3:
        st.subheader("🔍 財務體質")
        health_data = []
        for t in tickers_list:
            info = adv_data.get(t, {})
            is_etf = info.get('Type') == 'ETF'
            peg = info.get('PEG'); peg_s = "ETF" if is_etf else (f"{peg:.2f}" if peg is not None else "-")
            roe = info.get('ROE'); roe_s = "ETF" if is_etf else (f"{roe*100:.1f}%" if roe is not None else "-")
            pm = info.get('Profit_Margin'); pm_s = "ETF" if is_etf else (f"{pm*100:.1f}%" if pm is not None else "-")
            health_data.append({"代號": t, "PEG": peg_s, "ROE": roe_s, "淨利率": pm_s, "流動比": info.get('Current_Ratio'), "負債/權益": info.get('Debt_Equity')})
        st.dataframe(pd.DataFrame(health_data), use_container_width=True)
    with t4:
        st.subheader("🚦 回測")
        for t in tickers_list:
            if t in df_close.columns:
                s, b = run_traffic_light(df_close[t])
                if s is not None: st.line_chart(pd.concat([s, b], axis=1))
    with t5:
        st.subheader("💰 CFO")
        c1,c2 = st.columns(2)
        inc=c1.number_input("月收",80000); exp=c1.number_input("月支",40000)
        c1.metric("儲蓄率", f"{(inc-exp)/inc:.1%}")
        ast=c2.number_input("資產",15000000); lia=c2.number_input("負債",8000000)
        c2.metric("淨值", f"${ast-lia:,.0f}")
    with t6:
        st.subheader("🏠 房貸")
        amt=st.number_input("貸",10000000); rt=st.number_input("率",2.2)
        pmt,_=calc_mortgage(amt,30,rt)
        st.metric("月付", f"${pmt:,.0f}")

    # === TAB 7: 策略實驗室 (NEW) ===
    with t7:
        st.subheader("📈 買入賣出策略實驗室 (Strategy Lab)")
        st.info("💡 模擬情境：過去300天，初始本金$10,000，每個月1號發薪水再存入$10,000。嚴格執行 CFO V3 策略 vs 無腦定投。")
        
        lab_ticker = st.selectbox("選擇回測標的", tickers_list)
        
        if lab_ticker in df_close.columns:
            # 執行三種頻率的回測
            res_1d, roi_1d, strat_roi_1d, inv_1d, end_dca, end_1d = run_strategy_backtest(df_close[lab_ticker].to_frame(name='Close'), frequency_days=1)
            res_3d, roi_3d, strat_roi_3d, inv_3d, _, end_3d = run_strategy_backtest(df_close[lab_ticker].to_frame(name='Close'), frequency_days=3)
            res_7d, roi_7d, strat_roi_7d, inv_7d, _, end_7d = run_strategy_backtest(df_close[lab_ticker].to_frame(name='Close'), frequency_days=7)
            
            # 顯示結果 Metrics
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("總投入本金", f"${inv_1d:,.0f}")
            k2.metric("無腦定投 (DCA) 淨值", f"${end_dca:,.0f}", f"ROI: {roi_1d:.1%}")
            
            # 策略比較表
            strat_data = [
                {"策略頻率": "每日一次 (Daily)", "最終淨值": f"${end_1d:,.0f}", "總報酬率 (ROI)": f"{strat_roi_1d:.1%}", "本益比 (Profit/Cost)": f"{(end_1d-inv_1d)/inv_1d:.2f}"},
                {"策略頻率": "每3天一次 (3-Day)", "最終淨值": f"${end_3d:,.0f}", "總報酬率 (ROI)": f"{strat_roi_3d:.1%}", "本益比 (Profit/Cost)": f"{(end_3d-inv_3d)/inv_3d:.2f}"},
                {"策略頻率": "每週一次 (Weekly)", "最終淨值": f"${end_7d:,.0f}", "總報酬率 (ROI)": f"{strat_roi_7d:.1%}", "本益比 (Profit/Cost)": f"{(end_7d-inv_7d)/inv_7d:.2f}"},
            ]
            st.table(pd.DataFrame(strat_data))
            
            # 畫圖 (比較 1D 策略 vs DCA)
            st.markdown("#### 📊 資金曲線對決 (Strategy vs DCA)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res_1d.index, y=res_1d['DCA_Value'], name='無腦定投 (DCA)', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=res_1d.index, y=res_1d['Strat_Value'], name='CFO 策略 (Daily)', line=dict(color='#00BFFF', width=2)))
            fig.add_trace(go.Scatter(x=res_1d.index, y=res_1d['Invested'], name='投入本金', line=dict(color='green', width=1)))
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()