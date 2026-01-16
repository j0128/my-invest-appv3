import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ==============================================================================
# 0. 全局環境與拓撲常數設定 (Global Configuration & Topological Constants)
# ==============================================================================

# 拓撲常數 (Derived from Posa Lab Experiments)
# 這些參數來自於對 2021-2025 年市場數據的拓撲撕裂測試
TOPO_CONSTANTS = {
    "LIQUIDITY_THRESHOLD": -0.137,  # 最佳防禦閾值 (Trillion USD, 20-day change)
    "LAG_DAYS_TECH": 15,            # 科技股反應時滯 (Days)
    "LAG_DAYS_CRYPTO": 0,           # 加密貨幣無時滯 (Immediate Tear)
    "KELLY_LOOKBACK": 60,           # 動態凱利窗口 (對應 Q1 2026 週期)
    "RF_TREES": 100                 # 隨機森林樹數量
}

# 資產分類學 (Topological Taxonomy)
# 用於決定防禦模式 (Hard vs Soft Defense)
ASSET_TAXONOMY = {
    "Growth": ['BTC-USD', 'ETH-USD', 'ARKK', 'PLTR', 'NVDA', 'AMD', 'TSLA', 'TQQQ', 'SOXL'],
    "Defensive": ['KO', 'MCD', 'JNJ', 'PG', '2330.TW', 'SPY', 'TLT', 'GLD', 'SCHD']
}

st.set_page_config(
    page_title="Alpha 13.999: 拓撲指揮官 (Ultimate)",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# 注入 CSS 樣式 (Simplicial Complex Visualization)
st.markdown("""
<style>
    /* Metric Card - 深色儀表板風格 */
    .metric-card {
        background-color: #0E1117;
        border: 1px solid #444;
        border-radius: 5px;
        padding: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    /* 狀態標籤 */
    .bull-mode {
        color: #00FF7F; font-weight: bold; border: 1px solid #00FF7F; 
        padding: 2px 8px; border-radius: 4px; font-size: 0.9em;
    }
    .bear-mode {
        color: #FF4B4B; font-weight: bold; border: 1px solid #FF4B4B; 
        padding: 2px 8px; border-radius: 4px; font-size: 0.9em;
    }
    .defensive-tag {
        color: #FFD700; font-weight: bold; border: 1px solid #FFD700; 
        padding: 2px 8px; border-radius: 4px; font-size: 0.8em;
    }
    /* Tab 優化 */
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        height: 45px; background-color: #1E1E1E; border-radius: 5px 5px 0 0; color: #AAA;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00BFFF; color: #000; font-weight: bold;
    }
    /* 表格優化 */
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th {
        background-color: #262730;
        color: white;
    }
    td {
        border-bottom: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. 核心數據引擎 (Data Sheaf Engine)
# ==============================================================================

@st.cache_data(ttl=1800)
def fetch_market_data(tickers):
    """
    [Data Sheaf] 獲取微觀價格流形 (Micro Price Manifold)。
    包含使用者持倉與基準指數 (Benchmarks)。
    """
    benchmarks = ['SPY', 'QQQ', 'QLD', 'TQQQ', '^VIX', '^TNX', '^IRX', 'HYG', 'GC=F', 'HG=F', 'DX-Y.NYB'] 
    all_tickers = list(set(tickers + benchmarks))
    
    # 初始化數據容器
    data = {col: {} for col in ['Close', 'Open', 'High', 'Low', 'Volume']}
    
    for t in all_tickers:
        try:
            # 下載過去 2 年數據 (保證足夠計算 SMA200 與 60d Kelly)
            df = yf.Ticker(t).history(period="2y", auto_adjust=True)
            if df.empty: continue
            
            data['Close'][t] = df['Close']
            data['Open'][t] = df['Open']
            data['High'][t] = df['High']
            data['Low'][t] = df['Low']
            data['Volume'][t] = df['Volume']
        except Exception: 
            continue
            
    # 使用 ffill 填補缺失值，確保拓撲連續性
    return (
        pd.DataFrame(data['Close']).ffill(), 
        pd.DataFrame(data['High']).ffill(), 
        pd.DataFrame(data['Low']).ffill(), 
        pd.DataFrame(data['Volume']).ffill()
    )

@st.cache_data(ttl=3600*12)
def fetch_fred_macro(api_key):
    """
    [Global Section] 獲取宏觀全域截面數據。
    計算 Net Liquidity (Fed Assets - TGA - RRP)。
    """
    if not api_key: return None, None
    try:
        fred = Fred(api_key=api_key)
        
        # 關鍵因子
        walcl = fred.get_series('WALCL', observation_start='2023-01-01')   # Fed Assets
        tga = fred.get_series('WTREGEN', observation_start='2023-01-01')   # Treasury General Account
        rrp = fred.get_series('RRPONTSYD', observation_start='2023-01-01') # Reverse Repo
        fed_rate = fred.get_series('FEDFUNDS', observation_start='2023-01-01')
        
        # 構建 DataFrame 並對齊
        df = pd.DataFrame({'WALCL': walcl, 'TGA': tga, 'RRP': rrp}).ffill().dropna()
        
        # 計算淨流動性 (單位: Trillion)
        df['Net_Liquidity'] = (df['WALCL'] - df['TGA'] - df['RRP']) / 1000 
        
        # 利率日線化
        df_rate = pd.DataFrame({'Fed_Rate': fed_rate}).resample('D').ffill()
        
        return df, df_rate
    except Exception as e: 
        # st.sidebar.error(f"FRED API Error: {str(e)}") # 暫時隱藏錯誤，避免干擾
        return None, None

@st.cache_data(ttl=3600*24)
def get_advanced_info(ticker):
    """
    [Fundamental Sheaf] 獲取基本面元數據。
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            'Type': 'ETF' if 'ETF' in info.get('quoteType', '').upper() else 'Stock',
            'Target_Mean': info.get('targetMeanPrice'), 
            'PEG': info.get('pegRatio'),
            'Inst_Held': info.get('heldPercentInstitutions'), 
            'Short_Ratio': info.get('shortRatio'), 
            'ROE': info.get('returnOnEquity'),
            'Profit_Margin': info.get('profitMargins'),
            'Sector': info.get('sector', 'Unknown'),
            'Beta': info.get('beta', 1.0)
        }
    except: return {}


# ==============================================================================
# 2. 戰略模型與演算法 (Strategic Algorithms)
# ==============================================================================

def train_rf_model(df_close, ticker, days_forecast=30):
    """
    [Non-Linearity] 隨機森林預測模型。
    捕捉價格流形上的非線性特徵。
    """
    try:
        if ticker not in df_close.columns: return None
        
        df = pd.DataFrame({'Close': df_close[ticker]})
        df['Ret'] = df['Close'].pct_change()
        df['Vol'] = df['Ret'].rolling(20).std()
        df['SMA'] = df['Close'].rolling(20).mean() # Feature Engineering
        df['Target'] = df['Close'].shift(-days_forecast) # 預測未來
        df = df.dropna()
        
        if len(df) < 60: return None
        
        X = df.drop(columns=['Target', 'Close'])
        y = df['Target']
        
        model = RandomForestRegressor(n_estimators=TOPO_CONSTANTS['RF_TREES'], max_depth=5, random_state=42)
        model.fit(X, y)
        
        return model.predict(X.iloc[[-1]])[0]
    except: return None

def calc_targets_composite(ticker, df_close, df_high, df_low, f_data, days_forecast=30):
    """
    [Valuation Sheaf] 綜合估值體系。
    整合 ATR (波動), MC (漂移), Fib (結構), RF (AI), Consensus (基本面)。
    """
    if ticker not in df_close.columns: return None
    c = df_close[ticker]; h = df_high[ticker]; l = df_low[ticker]
    try:
        # 1. ATR Target (波動率邊界)
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        atr_val = tr.rolling(14).mean().iloc[-1]
        t_atr = c.iloc[-1] + (atr_val * np.sqrt(days_forecast))
        
        # 2. Monte Carlo (慣性漂移)
        mu = c.pct_change().mean()
        t_mc = c.iloc[-1] * ((1 + mu)**days_forecast)
        
        # 3. Fibonacci (結構阻力)
        recent = c.iloc[-60:]
        t_fib = recent.max() + (recent.max() - recent.min()) * 0.618 
        
        # 4. RF Model (AI)
        t_rf = train_rf_model(df_close, ticker, days_forecast)
        
        # 5. Analyst Target
        t_fund = f_data.get('Target_Mean')

        # 聚合
        targets = [t for t in [t_atr, t_mc, t_fib, t_fund, t_rf] if t is not None and not pd.isna(t)]
        t_avg = sum(targets)/len(targets) if targets else None
        
        return {"Avg": t_avg, "ATR": t_atr, "MC": t_mc, "Fib": t_fib, "RF": t_rf}
    except: return None

def run_backtest_lab_v2(ticker, df_close, df_high, df_low, df_macro, f_data, days_ago=30):
    """
    [V2 Radar Backtest] 含宏觀修正的個股回測。
    核心邏輯：若 30 天前流動性緊縮 (Threshold < -0.137T)，則強制下修預測值。
    """
    if ticker not in df_close.columns or len(df_close) < 250: return None
    
    # 定位時空坐標
    idx_past = len(df_close) - days_ago - 1
    date_past = df_close.index[idx_past]
    p_past = df_close[ticker].iloc[idx_past]
    p_now = df_close[ticker].iloc[-1]
    
    # 宏觀狀態檢查
    macro_status = "⚪ 中性"
    is_contraction = False
    
    if df_macro is not None and not df_macro.empty:
        try:
            m_idx = df_macro.index.get_indexer([date_past], method='ffill')[0]
            if m_idx > 20:
                liq_curr = df_macro['Net_Liquidity'].iloc[m_idx]
                liq_prev = df_macro['Net_Liquidity'].iloc[m_idx - 20]
                liq_chg = liq_curr - liq_prev
                
                # 應用實驗參數 -0.137
                if liq_chg < TOPO_CONSTANTS['LIQUIDITY_THRESHOLD']: 
                    is_contraction = True
                    macro_status = "🔻 緊縮 (Risk-Off)"
                elif liq_chg > 0.05:
                    macro_status = "💧 寬鬆 (Risk-On)"
        except: pass

    # 原始預測
    df_p = df_close.iloc[:idx_past+1]; h_p = df_high.iloc[:idx_past+1]; l_p = df_low.iloc[:idx_past+1]
    raw_targets = calc_targets_composite(ticker, df_p, h_p, l_p, f_data, days_ago)
    final_pred = raw_targets['Avg'] if raw_targets else None
    
    # 拓撲修正 (Topological Correction)
    note = ""
    if is_contraction and final_pred and final_pred > p_past:
        final_pred = final_pred * 0.85 # 強制下修 (實驗係數)
        note = "(因緊縮下修)"

    # 計算誤差
    if final_pred:
        err = (final_pred - p_now) / p_now
        process = f"[{macro_status}] 預測: {final_pred:.2f} {note} vs 現價: {p_now:.2f} | 誤差: {err:.1%}"
    else:
        err = 0; process = "N/A"
        
    return {"Past_Pred": final_pred, "Present_Value": p_now, "Error": err, "Process": process}

def calc_dynamic_kelly(series, lookback=60):
    """
    [Dynamic Kelly] 基於局部同調 (Local Homology) 的動態槓桿。
    $$ f = W - (1-W)/R $$
    """
    try:
        rets = series.iloc[-lookback:].pct_change().dropna()
        wins = rets[rets > 0]; losses = rets[rets < 0]
        
        if len(losses) == 0: return 0.5 # 全勝時限制
        win_rate = len(wins) / len(rets)
        avg_win = wins.mean(); avg_loss = abs(losses.mean())
        if avg_loss == 0: return 0.5
        
        kelly = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        return max(0.0, min(1.0, kelly * 0.5)) # Half-Kelly
    except: return 0.0

def analyze_trend_multi(series):
    """ 多重趨勢狀態判定 """
    if len(series) < 200: return {"status": "N/A", "p_now": series.iloc[-1], "is_bull": False}
    p = series.iloc[-1]; sma200 = series.rolling(200).mean().iloc[-1]
    sma200_prev = series.rolling(200).mean().iloc[-10]
    is_bull = (p > sma200) and (sma200 > sma200_prev)
    return {"status": "🔥 多頭" if p > sma200 else "🛑 空頭", "p_now": p, "is_bull": is_bull}

def calc_tech_indicators(series, vol_series):
    """ RSI, Slope, Volume Ratio """
    if len(series) < 60: return 50, 0, 1
    delta = series.diff()
    up = delta.clip(lower=0); down = -1 * delta.clip(upper=0)
    rs = up.ewm(com=13).mean() / down.ewm(com=13).mean()
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    ma20 = series.rolling(20).mean()
    slope = (ma20.iloc[-1] - ma20.iloc[-5]) / ma20.iloc[-5]
    
    vol_ma = vol_series.rolling(20).mean().iloc[-1]
    vr = vol_series.iloc[-1] / vol_ma if vol_ma > 0 else 1.0
    return rsi, slope, vr

def calc_six_dim_state(series):
    """ 六維狀態判定 (State Space Mapping) """
    if len(series) < 22: return "N/A"
    p = series.iloc[-1]; ma20 = series.rolling(20).mean().iloc[-1]; std = series.rolling(20).std().iloc[-1]
    
    if p > ma20 + 2*std * 1.05: return "H3 極限噴出"
    if p > ma20 + 2*std: return "H2 情緒過熱"
    if p > ma20: return "H1 多頭回歸"
    if p < ma20 - 2*std * 1.05: return "L3 恐慌崩盤"
    if p < ma20 - 2*std: return "L2 超賣區"
    return "L1 震盪整理"

# [RESTORED] 補回 calc_mvrv_z 函數
def calc_mvrv_z(series):
    """
    MVRV-Z Score 近似值 (用於判斷是否偏離均值過遠)。
    計算公式: (Price - SMA200) / Std200
    """
    if len(series) < 200: return None
    sma200 = series.rolling(200).mean()
    std200 = series.rolling(200).std()
    # 避免除以零
    z_score = (series - sma200) / (std200 + 1e-9)
    return z_score

def get_cfo_directive_v4(p_now, six_state, trend_status, bull_mode, rsi, slope, vol_ratio, mvrv_z, range_high, range_low):
    """ CFO 決策核心 V4 """
    if "L" in six_state and "空頭" in trend_status: return "⬛ 趨勢損毀 (清倉)", 0.0
    
    rsi_lim = 85 if bull_mode else 80
    if ("H3" in six_state) or (rsi > rsi_lim): return "🟥 極限噴出 (賣1/2)", 0.5
    if not bull_mode and range_high > 0 and p_now >= range_high: return "🟥 達預測高點 (賣1/2)", 0.5
    
    buy_signals = []; build_pct = 0.5 if bull_mode else 0.0
    if (mvrv_z is not None and mvrv_z < -0.5): buy_signals.append("🔵 價值買點")
    if "L2" in six_state: buy_signals.append("💎 抄底機會")
    if "多頭" in trend_status:
        if slope > 0.01: buy_signals.append("🔥 加速進攻"); build_pct = 0.8
        else: buy_signals.append("🟢 多頭確立"); build_pct = 0.5
        
    return (" | ".join(buy_signals) if buy_signals else ("🦁 牛市持倉" if bull_mode else "⬜ 觀望/持有")), build_pct

# [RESTORED] 補回 calc_obv_trend 函數
def calc_obv_trend(close, volume, lookback=20):
    try:
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        if len(obv) < lookback: return "N/A"
        delta = obv.iloc[-1] - obv.iloc[-lookback]
        return "🔥 吸籌" if delta > 0 else "🔻 出貨"
    except: return "N/A"

# [RESTORED] 補回 calc_obv 函數
def calc_obv(close, volume):
    if volume is None: return None
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

# [RESTORED] 補回 compare_with_leverage 函數
def compare_with_leverage(ticker, df_close):
    if ticker not in df_close.columns: return None
    benchs = ['QQQ', 'QLD', 'TQQQ']
    valid_benchs = [b for b in benchs if b in df_close.columns]
    if not valid_benchs: return None
    lookback = 252 if len(df_close) > 252 else len(df_close)
    df_compare = df_close[[ticker] + valid_benchs].iloc[-lookback:].copy()
    df_norm = df_compare / df_compare.iloc[0] * 100
    ret_ticker = df_norm[ticker].iloc[-1] - 100
    ret_tqqq = df_norm['TQQQ'].iloc[-1] - 100 if 'TQQQ' in df_norm else 0
    status = "👑 跑贏 TQQQ" if ret_ticker > ret_tqqq else "💀 輸給 TQQQ"
    return df_norm, status, ret_ticker, ret_tqqq


# ==============================================================================
# 3. 財務計算工具 (Financial Calculators)
# ==============================================================================

def calc_coast_fire(age, r_age, nw, save, ret, infl):
    years = r_age - age; real_ret = (1 + ret/100)/(1 + infl/100) - 1
    data = []; bal = nw; goal = save * 12 * 25
    for i in range(years + 1):
        data.append({"Age": age+i, "Balance": bal, "Fire_Goal": goal})
        bal = bal * (1 + real_ret) + save * 12
    return pd.DataFrame(data), bal

def calc_mortgage_advanced(princ, rate, years, extra):
    r = rate/100/12; n = years*12
    pmt = princ * (r*(1+r)**n)/((1+r)**n-1) if r > 0 else princ/n
    tot_int_n = pmt*n - princ
    bal = princ; tot_int_a = 0; m = 0
    while bal > 0 and m < n*2:
        inte = bal * r; paid = pmt - inte + extra
        if bal < paid: paid = bal
        bal -= paid; tot_int_a += inte; m += 1
    return pmt, tot_int_n, tot_int_a, tot_int_n - tot_int_a, (n - m)/12


# ==============================================================================
# 4. 回測實驗室 V3 (Final Topological Defensive Backtest)
# ==============================================================================

def run_strategy_backtest_salary_flow_v3(df_in, vol_in, df_macro, ticker_type="Growth"):
    """
    [V3 Final] 拓撲防禦回測。
    結合 A/B/C 實驗參數：
    - 閾值: -0.137T
    - 防禦分層: Growth (Hard) vs Defensive (Soft)
    """
    df = df_in.copy(); df['Volume'] = vol_in
    if len(df) > 500: df = df.iloc[-500:]
    
    # 宏觀數據對齊
    if df_macro is not None and not df_macro.empty:
        # 計算 20日 Net Liquidity 變化
        macro_sig = df_macro['Net_Liquidity'].diff(20).reindex(df.index).ffill()
    else:
        macro_sig = pd.Series(0, index=df.index)
        
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    # RSI
    delta = df['Close'].diff(); up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
    df['RSI'] = 100 - (100/(1 + up.ewm(13).mean()/down.ewm(13).mean()))
    
    cash_d = 0; stock_d = 0; cash_s = 0; stock_s = 0; inv = 0
    hist = []; last_m = -1
    
    for i in range(len(df)):
        p = df['Close'].iloc[i]; d = df.index[i]
        try: liq_trend = macro_sig.iloc[i]
        except: liq_trend = 0
        
        # DCA (發薪日)
        if d.month != last_m:
            cash_d += 10000; cash_s += 10000; inv += 10000; last_m = d.month
            buy = cash_d // p; stock_d += buy; cash_d -= buy * p
            
        if i > 20:
            ma20 = df['SMA20'].iloc[i]; ma200 = df['SMA200'].iloc[i]; rsi = df['RSI'].iloc[i]
            
            # --- 拓撲決策 (Topological Decision) ---
            # 1. 偵測全域截面狀態 (H0 Check)
            is_crunch = liq_trend < TOPO_CONSTANTS['LIQUIDITY_THRESHOLD']
            
            # 2. 分類防禦 (Stratified Defense)
            if is_crunch:
                if ticker_type in ["Growth", "Crypto", "High_Beta"]:
                    risk_mode = "HARD_DEFENSE" # 拓撲撕裂高風險 -> 空手
                else:
                    risk_mode = "SOFT_DEFENSE" # 拓撲穩定 -> 減半
            else:
                risk_mode = "NORMAL"
            
            # 3. 賣出執行
            sell = 0
            if risk_mode == "HARD_DEFENSE":
                sell = 1.0 # 強制清倉
            elif risk_mode == "SOFT_DEFENSE":
                if p < ma20: sell = 0.5 # 破月線減碼
            else:
                # 正常模式
                if p < ma20 and p < ma200: sell = 1.0
                elif rsi > 80: sell = 0.5
                
            if sell > 0 and stock_s > 0:
                s_amt = int(stock_s * sell); stock_s -= s_amt; cash_s += s_amt * p
                
            # 4. 買入執行 (Veto Power)
            if sell == 0:
                can_buy = True
                if risk_mode == "HARD_DEFENSE": can_buy = False # 危機時禁止買入成長股
                
                if can_buy:
                    bull = p > ma200
                    alloc = 0.8 if bull else 0.2
                    if cash_s > 100:
                        b_amt = cash_s * alloc // p; stock_s += b_amt; cash_s -= b_amt * p
                    
        hist.append({"Date": d, "DCA": cash_d + stock_d*p, "Strat": cash_s + stock_s*p})
        
    res = pd.DataFrame(hist).set_index("Date")
    final_d = (res['DCA'].iloc[-1]-inv)/inv if inv > 0 else 0
    final_s = (res['Strat'].iloc[-1]-inv)/inv if inv > 0 else 0
    return res, final_d, final_s, inv

def run_traffic_light(series):
    sma200 = series.rolling(200).mean()
    df = pd.DataFrame({'Close': series, 'SMA200': sma200})
    df['Signal'] = np.where(df['Close'] > df['SMA200'], 1, 0)
    df['Strat'] = (1 + df['Close'].pct_change() * df['Signal'].shift(1)).cumprod()
    df['BH'] = (1 + df['Close'].pct_change()).cumprod()
    return df['Strat'], df['BH']

def parse_input(text):
    port = {}
    for line in text.strip().split('\n'):
        if ',' in line:
            parts = line.split(',')
            try: port[parts[0].strip().upper()] = float(parts[1].strip())
            except: pass
    return port

# ==============================================================================
# 5. [NEW] 內建實驗套件 (In-App Experiment Suite)
# ==============================================================================
def run_in_app_experiment(prices, macro):
    """
    將 Colab 的 A/B/C 實驗封裝為 App 內功能。
    """
    st.markdown("### 🧪 實驗 C: 最佳閾值掃描 (Sensitivity Sweep)")
    
    # 簡化版實驗 C (針對 BTC)
    target = 'BTC-USD'
    if target in prices.columns:
        thresholds = np.linspace(-0.2, 0.0, 20)
        metrics = []
        
        df_base = pd.DataFrame({'Close': prices[target]})
        # 需確保 macro 有 Liq_Change
        df_base['Liq_Chg'] = macro['Net_Liquidity'].diff(20).reindex(df_base.index).ffill()
        df_base['Ret_BH'] = df_base['Close'].pct_change()
        
        progress = st.progress(0)
        for i, th in enumerate(thresholds):
            df = df_base.copy()
            # 策略：低於閾值空手
            df['Signal'] = np.where(df['Liq_Chg'] < th, 0, 1)
            df['Ret_Strat'] = df['Ret_BH'] * df['Signal'].shift(1)
            
            cum = (1 + df['Ret_Strat']).cumprod().iloc[-1] - 1
            vol = df['Ret_Strat'].std() * np.sqrt(252)
            sharpe = cum / vol if vol > 0 else 0
            metrics.append({'Threshold': th, 'Sharpe': sharpe})
            progress.progress((i+1)/len(thresholds))
            
        res_df = pd.DataFrame(metrics)
        best_th = res_df.loc[res_df['Sharpe'].idxmax()]['Threshold']
        
        st.success(f"🏆 計算完成！最佳防禦閾值: {best_th:.3f} T (目前設定: {TOPO_CONSTANTS['LIQUIDITY_THRESHOLD']} T)")
        
        fig = px.line(res_df, x='Threshold', y='Sharpe', title=f"{target} Sharpe Ratio vs Liquidity Threshold")
        fig.add_vline(x=best_th, line_dash="dash", line_color="green")
        st.plotly_chart(fig)
    else:
        st.warning("需持有 BTC-USD 才能執行此實驗。")


# ==============================================================================
# 6. 主程式 (Main Application)
# ==============================================================================

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ 指揮系統設定 (Ultimate)")
        fred_key = st.secrets.get("FRED_API_KEY", st.text_input("FRED API Key", type="password"))
        
        user_cash = st.number_input("💰 現金儲備 (USD)", value=10000.0, step=1000.0)
        user_input = st.text_area("持倉清單", "BTC-USD, 10000\nNVDA, 10000\n2330.TW, 10000\nKO, 5000", height=150)
        
        p_dict = parse_input(user_input); tickers = list(p_dict.keys())
        st.metric("🏦 總資產", f"${(user_cash + sum(p_dict.values())):,.0f}")
        
        if st.button("🚀 啟動 Alpha 指揮中心", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False):
        st.info("請輸入資料並啟動。")
        return

    # --- Data Fetching ---
    with st.spinner("🦅 Alpha 13.999 正在執行全域拓撲掃描..."):
        df_close, df_high, df_low, df_vol = fetch_market_data(tickers)
        df_macro, df_fed = fetch_fred_macro(fred_key)
        adv_data = {t: get_advanced_info(t) for t in tickers}

    # --- Tabs ---
    tabs = st.tabs([
        "🦅 戰略戰情", "🐋 籌碼", "🔍 體質", "🚦 回測", 
        "💰 CFO", "🏠 房貸", "📊 實驗室", "🧪 拓撲驗證"
    ])
    
    t1, t2, t3, t4, t5, t6, t7, t8 = tabs

    # === TAB 1: 宏觀戰情 (RESTORED TABLE) ===
    with t1:
        st.title("🦅 Alpha 13.999: 混合戰略指揮中心")
        
        # 1. Macro Dashboard
        if df_macro is not None:
            liq = df_macro['Net_Liquidity'].iloc[-1]
            try: liq_chg = liq - df_macro['Net_Liquidity'].iloc[-20] # 20天變化
            except: liq_chg = 0
            
            # 狀態判定
            is_crunch = liq_chg < TOPO_CONSTANTS['LIQUIDITY_THRESHOLD']
            status_html = f'<span class="bear-mode">拓撲撕裂 (HARD DEFENSE)</span>' if is_crunch else f'<span class="bull-mode">流動性安全</span>'
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("💧 淨流動性", f"${liq:.2f}T", f"{liq_chg:+.3f}T (20d)")
            c2.markdown(f"**全域狀態**: {status_html}", unsafe_allow_html=True)
            if is_crunch: st.error(f"⚠️ 警告：觸發縮表閾值 ({TOPO_CONSTANTS['LIQUIDITY_THRESHOLD']}T)！啟動防禦模式。")
        else:
            st.warning("⚠️ 無 FRED 數據，宏觀功能失效。")
            liq_chg = 0

        # VIX & Rates
        vix = df_close['^VIX'].iloc[-1] if '^VIX' in df_close.columns else 0
        fed = df_fed['Fed_Rate'].iloc[-1] if df_fed is not None else 0
        k1, k2, k3 = st.columns(3)
        k1.metric("🌪️ VIX", f"{vix:.2f}")
        k2.metric("🏦 Fed利率", f"{fed:.2f}%")

        # 2. Strategy Table (RESTORED HTML VERSION)
        st.markdown("#### 📊 CFO 混合戰略總表 (含 $±2\sigma$ 預測範圍)")
        summary = []
        for t in tickers:
            if t not in df_close.columns: continue
            
            # Trend & Indicators
            tr = analyze_trend_multi(df_close[t])
            rsi, slope, vr = calc_tech_indicators(df_close[t], df_vol[t])
            six = calc_six_dim_state(df_close[t])
            
            # [CRITICAL] 確保 mvrv 安全計算
            try:
                mvrv = calc_mvrv_z(df_close[t]).iloc[-1] 
            except:
                mvrv = 0
            
            # Targets & Backtest
            targets = calc_targets_composite(t, df_close, df_high, df_low, adv_data.get(t,{}), 30)
            tgt_val = targets['Avg'] if targets else 0
            
            # Calculate Range
            try:
                vol_22 = df_close[t].pct_change().std() * np.sqrt(22)
                pred_range = f"${tr['p_now']*(1-2*vol_22):.2f} - ${tr['p_now']*(1+2*vol_22):.2f}"
            except: pred_range = "N/A"
            
            # V2 Radar (Injected Threshold)
            bt = run_backtest_lab_v2(t, df_close, df_high, df_low, df_macro, adv_data.get(t,{}), 30)
            
            # CFO Directive
            act, _ = get_cfo_directive_v4(tr['p_now'], six, tr['status'], tr['is_bull'], rsi, slope, vr, mvrv, tgt_val*1.05, tgt_val*0.95)
            
            mode_tag = f'<span class="bull-mode">BULL</span>' if tr['is_bull'] else f'<span class="bear-mode">BEAR</span>'
            
            summary.append({
                "代號": t, 
                "模式": mode_tag,
                "現價": f"${tr['p_now']:.2f}",
                "CFO 指令": act,
                "預期範圍(±2σ)": pred_range,
                "目標價(Avg)": f"${tgt_val:.2f}",
                "拓撲回測誤差": f"{bt['Error']:.1%}" if bt else "N/A"
            })
        
        # 使用 HTML 渲染多彩表格
        st.write(pd.DataFrame(summary).to_html(escape=False), unsafe_allow_html=True)
        
        # 顯示詳細回測文字
        with st.expander("🦅 點擊查看詳細回測修正邏輯"):
            for t in tickers:
                res = run_backtest_lab_v2(t, df_close, df_high, df_low, df_macro, adv_data.get(t,{}), 30)
                if res: st.text(f"{t}: {res['Process']}")

    # === TAB 2: 籌碼 ===
    with t2:
        st.subheader("🐋 動態凱利籌碼")
        c_data = []
        for t in tickers:
            if t not in df_close.columns: continue
            k = calc_dynamic_kelly(df_close[t], TOPO_CONSTANTS['KELLY_LOOKBACK'])
            obv = calc_obv_trend(df_close[t], df_vol[t])
            c_data.append({"代號": t, "動態凱利%": f"{k*100:.1f}%", "OBV": obv})
        st.dataframe(pd.DataFrame(c_data))

    # === TAB 3-6: 標準功能 ===
    with t3: st.dataframe(pd.DataFrame([{"代號": t, "ROE": adv_data.get(t,{}).get('ROE')} for t in tickers]))
    with t4: 
        for t in tickers: 
            if t in df_close.columns: st.line_chart(run_traffic_light(df_close[t])[0])

    with t5: 
        st.subheader("💰 CFO 財報")
        nw = st.number_input("淨資產", 2000000.0)
        if st.button("計算FIRE"):
            df_f, bal = calc_coast_fire(35, 60, nw, 30000, 7, 2)
            st.metric("預估資產", f"${bal:,.0f}")
            st.line_chart(df_f.set_index("Age")['Balance'])
            
    with t6:
        st.subheader("🏠 房貸試算")
        amt = st.number_input("貸款", 10000000.0)
        if st.button("計算房貸"):
            pmt, _, tot_a, sav, _ = calc_mortgage_advanced(amt, 2.2, 30, 5000)
            st.metric("月付", f"${pmt:,.0f}")
            st.metric("省息", f"${sav:,.0f}")

    # === TAB 7: 策略實驗室 (V3 Final) ===
    with t7:
        st.subheader("📊 拓撲實驗室 (V3 Final - 分類防禦)")
        st.info(f"當前防禦參數：閾值 {TOPO_CONSTANTS['LIQUIDITY_THRESHOLD']}T | 科技股時滯 {TOPO_CONSTANTS['LAG_DAYS_TECH']}天")
        
        lab_ticker = st.selectbox("回測標的", sorted(list(set(tickers + ['QQQ', 'SPY']))))
        
        # 自動分類
        if lab_ticker in ASSET_TAXONOMY['Growth']: t_type = "Growth"
        elif lab_ticker in ASSET_TAXONOMY['Defensive']: t_type = "Defensive"
        else: t_type = "Growth" # 預設高風險
        
        st.write(f"標的類型: **{t_type}** (若是 Growth 則觸發 Hard Defense)")
        
        if lab_ticker in df_close.columns:
            res, r_d, r_s, inv = run_strategy_backtest_salary_flow_v3(
                df_close[lab_ticker].to_frame(name='Close'), 
                df_vol[lab_ticker], 
                df_macro,
                ticker_type=t_type
            )
            
            c1, c2, c3 = st.columns(3)
            c1.metric("投入本金", f"${inv:,.0f}")
            c2.metric("DCA", f"{r_d:.1%}")
            c3.metric("拓撲策略", f"{r_s:.1%}", delta=f"{(r_s-r_d)*100:.1f}%")
            st.plotly_chart(px.line(res[['DCA', 'Strat']]))
            
            # [NEW] 顯示比較雷達圖
            if 'TQQQ' in df_close.columns:
                comp_res = compare_with_leverage(lab_ticker, df_close)
                if comp_res:
                    st.success(f"槓桿比較: {comp_res[1]}")

    # === TAB 8: 內建驗證 (NEW) ===
    with t8:
        st.subheader("🧪 拓撲驗證實驗室 (In-App)")
        st.write("在此執行即時參數掃描，驗證 -0.137T 是否仍為最佳解。")
        if st.button("執行實驗 C (敏感度掃描)"):
            if df_macro is not None:
                run_in_app_experiment(df_close, df_macro)
            else:
                st.error("需連接 FRED API 才能執行實驗。")

if __name__ == "__main__":
    main()