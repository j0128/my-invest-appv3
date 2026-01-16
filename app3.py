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

# ==========================================
# 0. 全局環境設定 (Global Configuration)
# ==========================================
st.set_page_config(
    page_title="Alpha 13.8: 拓撲指揮官",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# 注入 CSS 樣式，優化視覺層次 (Simplicial Complex Visualization)
st.markdown("""
<style>
    /* Metric Card 樣式 - 模擬深色模式儀表板 */
    .metric-card {
        background-color: #0E1117;
        border: 1px solid #444;
        border-radius: 5px;
        padding: 15px;
        color: white;
    }
    /* 牛市標籤 - 高亮綠色 */
    .bull-mode {
        color: #00FF7F;
        font-weight: bold;
        border: 1px solid #00FF7F;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.9em;
    }
    /* 熊市標籤 - 高亮紅色 */
    .bear-mode {
        color: #FF4B4B;
        font-weight: bold;
        border: 1px solid #FF4B4B;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.9em;
    }
    /* Tab 分頁樣式優化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1E1E1E;
        border-radius: 5px 5px 0 0;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00BFFF; /* 亮藍色代表選中狀態 */
        color: black;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 1. 核心數據引擎 (Data Sheaf Engine)
# ==========================================

@st.cache_data(ttl=1800)
def fetch_market_data(tickers):
    """
    獲取市場價格數據，構建基礎單純複形 (Base Simplicial Complex)。
    包含基準指數: SPY, QQQ, VIX, TNX (美債), IRX, HYG, 黃金, 銅, 美元指數。
    """
    benchmarks = ['SPY', 'QQQ', 'QLD', 'TQQQ', '^VIX', '^TNX', '^IRX', 'HYG', 'GC=F', 'HG=F', 'DX-Y.NYB'] 
    # 合併使用者自選與基準標的
    all_tickers = list(set(tickers + benchmarks))
    
    data = {col: {} for col in ['Close', 'Open', 'High', 'Low', 'Volume']}
    
    for t in all_tickers:
        try:
            # 下載過去 2 年數據，用於構建較長期的移動平均與趨勢
            df = yf.Ticker(t).history(period="2y", auto_adjust=True)
            if df.empty: continue
            
            data['Close'][t] = df['Close']
            data['Open'][t] = df['Open']
            data['High'][t] = df['High']
            data['Low'][t] = df['Low']
            data['Volume'][t] = df['Volume']
        except Exception as e:
            continue
            
    # 使用 ffill 處理缺失值，確保數據流形 (Data Manifold) 的連續性
    return (
        pd.DataFrame(data['Close']).ffill(), 
        pd.DataFrame(data['High']).ffill(), 
        pd.DataFrame(data['Low']).ffill(), 
        pd.DataFrame(data['Volume']).ffill()
    )

@st.cache_data(ttl=3600*12)
def fetch_fred_macro(api_key):
    """
    獲取宏觀經濟數據 (FRED)，用於計算淨流動性 (Net Liquidity)。
    Net Liquidity = WALCL (Fed資產) - TGA (財政部帳戶) - RRP (逆回購)
    這代表了市場 H0 (Global Section) 的支撐力量。
    """
    if not api_key: return None, None
    try:
        fred = Fred(api_key=api_key)
        
        # 獲取關鍵流動性因子
        walcl = fred.get_series('WALCL', observation_start='2024-01-01')
        tga = fred.get_series('WTREGEN', observation_start='2024-01-01')
        rrp = fred.get_series('RRPONTSYD', observation_start='2024-01-01')
        fed_rate = fred.get_series('FEDFUNDS', observation_start='2023-01-01')
        
        # 構建流動性 DataFrame
        df = pd.DataFrame({'WALCL': walcl, 'TGA': tga, 'RRP': rrp}).ffill().dropna()
        # 單位轉換為 Trillion (兆美元)
        df['Net_Liquidity'] = (df['WALCL'] - df['TGA'] - df['RRP']) / 1000 
        
        # 利率數據重採樣至日頻率
        df_rate = pd.DataFrame({'Fed_Rate': fed_rate}).resample('D').ffill()
        
        return df, df_rate
    except Exception: 
        return None, None

@st.cache_data(ttl=3600*24)
def get_advanced_info(ticker):
    """
    獲取個股深度基本面數據 (Fundamental Sheaf)。
    包含 PEG, 機構持股, 空單比率等。
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        peg = info.get('pegRatio')
        
        return {
            'Type': 'ETF' if 'ETF' in info.get('quoteType', '').upper() else 'Stock',
            'Target_Mean': info.get('targetMeanPrice'), 
            'PEG': peg,
            'Inst_Held': info.get('heldPercentInstitutions'), 
            'Insider_Held': info.get('heldPercentInsiders'),
            'Short_Ratio': info.get('shortRatio'), 
            'Current_Ratio': info.get('currentRatio'),
            'Debt_Equity': info.get('debtToEquity'), 
            'ROE': info.get('returnOnEquity'),
            'Profit_Margin': info.get('profitMargins')
        }
    except Exception: 
        return {}


# ==========================================
# 2. 戰略模型與演算法 (Strategic Algorithms)
# ==========================================

def train_rf_model(df_close, ticker, days_forecast=30):
    """
    隨機森林回歸模型 (Random Forest)。
    用於捕捉非線性價格特徵。
    """
    try:
        if ticker not in df_close.columns: return None
        
        df = pd.DataFrame(index=df_close.index)
        df['Close'] = df_close[ticker]
        df['Ret'] = df['Close'].pct_change()
        df['Vol'] = df['Ret'].rolling(20).std()
        df['SMA'] = df['Close'].rolling(20).mean()
        df['Target'] = df['Close'].shift(-days_forecast)
        
        df = df.dropna()
        if len(df) < 60: return None
        
        X = df.drop(columns=['Target', 'Close'])
        y = df['Target']
        
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X, y)
        
        # 預測當前特徵對應的未來價格
        return model.predict(X.iloc[[-1]])[0]
    except Exception: 
        return None

def calc_targets_composite(ticker, df_close, df_high, df_low, f_data, days_forecast=30):
    """
    綜合估值模型 (Valuation Sheaf)。
    結合 ATR, 蒙地卡羅(簡化版), 費波南希, 基本面目標價, 機器學習預測。
    """
    if ticker not in df_close.columns: return None
    c = df_close[ticker]; h = df_high[ticker]; l = df_low[ticker]
    try:
        # 1. ATR Target (波動率目標)
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        t_atr = c.iloc[-1] + (tr.rolling(14).mean().iloc[-1] * np.sqrt(days_forecast))
        
        # 2. Monte Carlo Logic (基於漂移率)
        mu = c.pct_change().mean()
        t_mc = c.iloc[-1] * ((1 + mu)**days_forecast)
        
        # 3. Fibonacci Extension (近期高點延伸)
        recent = c.iloc[-60:]
        t_fib = recent.max() + (recent.max() - recent.min()) * 0.618 
        
        # 4. Random Forest (AI 預測)
        t_rf = train_rf_model(df_close, ticker, days_forecast)
        
        # 5. Analysts Target (華爾街共識)
        t_fund = f_data.get('Target_Mean')

        # 聚合所有非空目標價
        targets = [t for t in [t_atr, t_mc, t_fib, t_fund, t_rf] if t is not None and not pd.isna(t)]
        t_avg = sum(targets) / len(targets) if targets else None
        
        return {"Avg": t_avg, "ATR": t_atr, "MC": t_mc, "Fib": t_fib, "RF": t_rf}
    except Exception: 
        return None

def run_backtest_lab_v2(ticker, df_close, df_high, df_low, df_macro, f_data, days_ago=30):
    """
    [UPDATED] V2 拓撲回測實驗室:
    引入宏觀流動性 (df_macro) 作為全域截面修正。
    驗證: "在考慮 Fed 流動性狀態下，30天前的預測是否準確?"
    """
    # 1. 基本檢查
    if ticker not in df_close.columns or len(df_close) < 250: return None
    
    # 2. 定位時空坐標 (30天前)
    idx_past = len(df_close) - days_ago - 1
    date_past = df_close.index[idx_past]
    p_past = df_close[ticker].iloc[idx_past]
    p_now = df_close[ticker].iloc[-1]
    
    # 3. 獲取當時的 "全域流動性狀態" (Global Section at t-30)
    # 這裡我們看過去 20 天的流動性變化
    macro_status = "⚪ 中性"
    is_contraction = False

    if df_macro is not None and not df_macro.empty:
        try:
            # 找到最接近 date_past 的宏觀數據
            macro_idx = df_macro.index.get_indexer([date_past], method='ffill')[0]
            if macro_idx > 20: # 確保有足夠歷史
                liq_current = df_macro['Net_Liquidity'].iloc[macro_idx]
                liq_prev = df_macro['Net_Liquidity'].iloc[macro_idx - 20]
                liq_change = liq_current - liq_prev
                
                is_contraction = liq_change < -0.05 # 縮表閾值 (例如減少500億)
                if is_contraction:
                    macro_status = "🔻 緊縮"
                elif liq_change > 0.05:
                    macro_status = "💧 寬鬆"
        except:
            pass

    # 4. 計算 "原始" 技術目標價 (Valuation Sheaf)
    df_past = df_close.iloc[:idx_past+1]
    h_past = df_high.iloc[:idx_past+1]
    l_past = df_low.iloc[:idx_past+1]
    
    raw_targets = calc_targets_composite(ticker, df_past, h_past, l_past, f_data, days_ago)
    raw_pred = raw_targets['Avg'] if raw_targets else None
    
    # 5. 應用 "拓撲修正" (Topological Correction)
    # 如果當時流動性在緊縮，模型不應該樂觀看漲。
    # 修正邏輯：如果縮表，將目標價強制修正為 "防禦性價格" (例如打9折)
    final_pred = raw_pred
    note = ""
    
    if is_contraction and raw_pred and raw_pred > p_past:
        # 拓撲矛盾：流動性收縮，但技術面看漲 -> 視為 "假突破" 風險
        final_pred = raw_pred * 0.9 # 強制下修預期
        note = "(因緊縮下修)"
        
    # 6. 計算誤差
    if final_pred:
        diff = final_pred - p_now
        err = diff / p_now
        calc_process = f"[{macro_status}] 預測: {final_pred:.2f} {note} vs 現價: {p_now:.2f} | 誤差: {err:.1%}"
    else:
        err = 0; calc_process = "N/A"
        
    return {
        "Past_Pred": final_pred, 
        "Present_Value": p_now, 
        "Error": err, 
        "Process": calc_process,
        "Macro_State": macro_status
    }

def analyze_trend_multi(series):
    """
    多重趨勢判定。
    定義牛/熊市狀態空間。
    """
    if len(series) < 200: return {"status": "資料不足", "p_now": series.iloc[-1], "is_bull": False}
    
    p_now = series.iloc[-1]
    sma200 = series.rolling(200).mean().iloc[-1]
    sma200_prev = series.rolling(200).mean().iloc[-10]
    
    # 牛市定義：價格在年線上，且年線斜率向上
    is_bull = (p_now > sma200) and (sma200 > sma200_prev)
    status = "🔥 多頭" if p_now > sma200 else "🛑 空頭"
    
    return {"status": status, "p_now": p_now, "sma200": sma200, "is_bull": is_bull}

def calc_dynamic_kelly(series, lookback=60):
    """
    【核心更新】動態凱利準則 (Dynamic Kelly Formula)
    
    數學形式:
    $$ f^* = W - \frac{1-W}{R} $$
    其中:
    - W (Win Rate): 勝率 (過去 lookback 天)
    - R (Win/Loss Ratio): 盈虧比 (平均獲利 / 平均虧損)
    
    參數:
    - lookback: 強制設定為 60 天 (Q1 2026 戰術週期)
    
    返回:
    - 建議倉位比例 (0.0 ~ 1.0)，已應用 Half-Kelly 進行保守修正。
    """
    try:
        # 計算日收益率
        rets = series.iloc[-lookback:].pct_change().dropna()
        
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        
        # 極端情況處理
        if len(losses) == 0: return 0.5  # 全勝時期，限制最大倉位
        
        win_rate = len(wins) / len(rets)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        if avg_loss == 0: return 0.5
        
        win_loss_ratio = avg_win / avg_loss
        
        # 凱利公式
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # 應用 Half-Kelly 並限制範圍 [0, 1]
        return max(0.0, min(1.0, kelly * 0.5)) 
    except: 
        return 0.0

def calc_mvrv_z(series):
    """
    MVRV-Z Score 近似值 (用於判斷是否偏離均值過遠)。
    """
    if len(series) < 200: return None
    sma200 = series.rolling(200).mean()
    std200 = series.rolling(200).std()
    return (series - sma200) / std200

def calc_tech_indicators(series, vol_series):
    """
    技術指標計算：RSI, 斜率 (Slope), 量能比 (Volume Ratio)。
    """
    if len(series) < 60: return None, None, None
    
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    # RSI Calculation
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    # Slope Calculation (MA20)
    ma20 = series.rolling(20).mean()
    slope = (ma20.iloc[-1] - ma20.iloc[-5]) / ma20.iloc[-5]
    
    # Volume Ratio
    vol_ma = vol_series.rolling(20).mean().iloc[-1]
    vol_ratio = vol_series.iloc[-1] / vol_ma if vol_ma > 0 else 1.0
    
    return rsi, slope, vol_ratio

def calc_six_dim_state(series):
    """
    六維狀態判定 (Six-Dimensional State Space)。
    將價格位置映射到離散狀態集 {H3, H2, H1, L1, L2, L3}。
    """
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

def get_cfo_directive_v4(p_now, six_state, trend_status, bull_mode, rsi, slope, vol_ratio, mvrv_z, range_high, range_low):
    """
    CFO 指揮官決策邏輯 V4。
    綜合所有拓撲特徵，輸出具體操作指令與建議倉位。
    """
    # 1. 趨勢損毀檢查 (Stop Loss Condition)
    if "L" in six_state and "空頭" in trend_status: 
        return "⬛ 趨勢損毀 (清倉)", 0.0
    
    # 2. 過熱檢查 (Overheated)
    rsi_limit = 85 if bull_mode else 80
    if ("H3" in six_state) or (rsi is not None and rsi > rsi_limit): 
        return "🟥 極限噴出 (賣1/2)", 0.5
        
    if not bull_mode:
        if range_high > 0 and p_now >= range_high: return "🟥 達預測高點 (賣1/2)", 0.5
        if "H2" in six_state: return "🟧 過熱減碼 (賣1/3)", 0.66
        
    # 3. 買入/持有信號
    buy_signals = []
    build_pct = 0.5 if bull_mode else 0.0
    
    # 價值區檢查
    if (mvrv_z is not None and mvrv_z < -0.5) or (range_low > 0 and p_now < range_low): 
        buy_signals.append("🔵 價值買點")
        build_pct = max(build_pct, 0.5)
        
    # 技術性反彈/抄底
    if "L2" in six_state: 
        buy_signals.append("💎 抄底機會")
        build_pct = max(build_pct, 0.3)
        
    # 趨勢跟隨
    if "多頭" in trend_status:
        if slope is not None and slope > 0.01 and vol_ratio > 1.5: 
            buy_signals.append("🔥 加速進攻")
            build_pct = max(build_pct, 0.8)
        elif slope is not None and slope > 0: 
            buy_signals.append("🟢 多頭確立")
            build_pct = max(build_pct, 0.5)
        else: 
            buy_signals.append("🟢 轉強試單")
            build_pct = max(build_pct, 0.2)
            
    return (" | ".join(buy_signals) if buy_signals else ("🦁 牛市持倉" if bull_mode else "⬜ 觀望/持有")), build_pct

def calc_obv_trend(close, volume, lookback=20):
    """
    OBV (On-Balance Volume) 趨勢分析。
    判斷資金流向是否健康。
    """
    try:
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        if len(obv) < lookback: return "N/A"
        
        delta = obv.iloc[-1] - obv.iloc[-lookback]
        if delta > 0: return "🔥 吸籌 (買入)"
        else: return "🔻 出貨 (賣出)"
    except: 
        return "N/A"

def calc_obv(close, volume):
    if volume is None: return None
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

def compare_with_leverage(ticker, df_close):
    """
    槓桿 ETF 對比分析。
    """
    if ticker not in df_close.columns: return None
    benchs = ['QQQ', 'QLD', 'TQQQ']
    valid_benchs = [b for b in benchs if b in df_close.columns]
    if not valid_benchs: return None
    
    lookback = 252 if len(df_close) > 252 else len(df_close)
    df_compare = df_close[[ticker] + valid_benchs].iloc[-lookback:].copy()
    
    # 正規化比較 (歸一化為 100 起始)
    df_norm = df_compare / df_compare.iloc[0] * 100
    
    ret_ticker = df_norm[ticker].iloc[-1] - 100
    ret_tqqq = df_norm['TQQQ'].iloc[-1] - 100 if 'TQQQ' in df_norm else 0
    
    status = "👑 跑贏 TQQQ" if ret_ticker > ret_tqqq else "💀 輸給 TQQQ"
    return df_norm, status, ret_ticker, ret_tqqq


# ==========================================
# 3. 財務深度計算 (Financial Deep Calculation)
# ==========================================

def calc_coast_fire(current_age, retire_age, current_net_worth, monthly_saving, annual_return, inflation):
    """
    Coast FIRE 模擬計算。
    計算資產複利路徑。
    """
    years = retire_age - current_age
    real_return = (1 + annual_return/100) / (1 + inflation/100) - 1
    
    data = []
    balance = current_net_worth
    fire_number = (monthly_saving * 12 * 25) 
    
    for i in range(years + 1):
        age = current_age + i
        data.append({"Age": age, "Balance": balance, "Fire_Goal": fire_number})
        # 複利公式 + 年化儲蓄
        balance = balance * (1 + real_return) + (monthly_saving * 12)
        
    return pd.DataFrame(data), balance

def calc_mortgage_advanced(principal, rate, years, extra_monthly):
    """
    進階房貸計算器。
    支援額外還款 (Extra Payment) 對利息與年限的影響。
    """
    r = rate / 100 / 12
    n_months = years * 12
    
    # 標準月付公式
    if r > 0:
        monthly_payment = principal * (r * (1 + r)**n_months) / ((1 + r)**n_months - 1)
    else:
        monthly_payment = principal / n_months
    
    total_interest_normal = (monthly_payment * n_months) - principal
    
    balance = principal
    total_interest_acc = 0
    months_acc = 0
    
    # 模擬還款過程
    while balance > 0:
        interest = balance * r
        principal_paid = monthly_payment - interest + extra_monthly
        
        if balance < principal_paid:
            principal_paid = balance
            
        balance -= principal_paid
        total_interest_acc += interest
        months_acc += 1
        
        # 安全中止條件，避免無限迴圈
        if months_acc > n_months * 2: break
        
    saved_interest = total_interest_normal - total_interest_acc
    years_saved = (n_months - months_acc) / 12
    
    return monthly_payment, total_interest_normal, total_interest_acc, saved_interest, years_saved


# ==========================================
# 4. 回測實驗室 V3 (拓撲防禦版)
# ==========================================

def run_strategy_backtest_salary_flow_v3(df_in, vol_in, df_macro):
    """
    V3 拓撲增強版回測: 
    引入 FRED 宏觀流動性 (df_macro) 作為「全域過濾器 (Global Filter)」。
    當流動性收縮時，強制執行防禦策略。
    """
    df = df_in.copy()
    df['Volume'] = vol_in
    
    # --- 1. 數據對齊 (Data Alignment) ---
    # 將宏觀數據對齊到日線 (使用 ffill 避免前視偏誤)
    if df_macro is not None and not df_macro.empty:
        # 計算流動性趨勢 (20日變化)
        macro_signal = df_macro['Net_Liquidity'].diff(20).reindex(df.index).ffill()
    else:
        macro_signal = pd.Series(0, index=df.index) # 如果沒數據就設為中性

    if len(df) > 500: df = df.iloc[-500:] # 取近兩年
    # 確保 macro_signal 切片長度一致
    macro_signal = macro_signal.tail(len(df))

    # --- 2. 技術指標 ---
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['Upper'] = df['SMA20'] + 2 * df['Close'].rolling(20).std()
    
    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0).abs()
    df['RSI'] = 100 - (100 / (1 + up.ewm(13).mean() / down.ewm(13).mean()))
    
    # --- 3. 回測迴圈 ---
    cash_dca = 0; shares_dca = 0
    cash_strat = 0; shares_strat = 0
    invested = 0
    history = []
    last_month = -1
    
    for i in range(len(df)):
        p = df['Close'].iloc[i]
        date = df.index[i]
        
        # 獲取當下的流動性動能
        try:
            liq_trend = macro_signal.iloc[i]
        except:
            liq_trend = 0
        
        # 每月發薪日注入資金
        if date.month != last_month:
            cash_dca += 10000; cash_strat += 10000; invested += 10000
            last_month = date.month
            
            # DCA: 無腦買入
            buy_dca = cash_dca // p
            shares_dca += buy_dca
            cash_dca -= buy_dca * p
            
        if i > 20:
            ma20 = df['SMA20'].iloc[i]
            ma200 = df['SMA200'].iloc[i]
            rsi = df['RSI'].iloc[i]
            
            # --- 拓撲決策核心 (Topological Core) ---
            
            # 狀態 A: 全域流動性危機 (Global Crunch)
            # 條件: 流動性在收縮 (liq_trend < -0.05T)
            risk_mode = "DEFENSIVE" if liq_trend < -0.05 else "NORMAL"

            # 賣出邏輯
            sell_pct = 0
            if risk_mode == "DEFENSIVE":
                # 在流動性危機中，只要跌破月線就砍，絕不留戀
                if p < ma20: sell_pct = 1.0 
            else:
                # 正常模式：跌破年線或過熱才賣
                if p < ma20 and p < ma200: sell_pct = 1.0
                elif rsi > 80: sell_pct = 0.5

            # 執行賣出
            if sell_pct > 0 and shares_strat > 0:
                s_amt = int(shares_strat * sell_pct)
                shares_strat -= s_amt
                cash_strat += s_amt * p
                
            # 買入邏輯
            # 關鍵差異：如果處於 DEFENSIVE 模式，禁止買入 (Veto)
            if sell_pct == 0 and risk_mode == "NORMAL":
                bull = (p > ma200)
                buy_pct = 0.8 if bull else 0.2
                
                if cash_strat > 100:
                    b_val = cash_strat * buy_pct
                    buy = b_val // p
                    shares_strat += buy
                    cash_strat -= buy * p
                    
        history.append({
            "Date": date, 
            "DCA": cash_dca + shares_dca * p, 
            "Strat": cash_strat + shares_strat * p,
            "Liquidity_Trend": liq_trend
        })
        
    res = pd.DataFrame(history).set_index("Date")
    
    # 避免除以零錯誤
    if invested > 0:
        final_dca = (res['DCA'].iloc[-1]-invested)/invested
        final_strat = (res['Strat'].iloc[-1]-invested)/invested
    else:
        final_dca = 0; final_strat = 0
    
    return res, final_dca, final_strat, invested

def run_traffic_light(series):
    """
    SMA200 紅綠燈策略回測。
    """
    sma200 = series.rolling(200).mean()
    df = pd.DataFrame({'Close': series, 'SMA200': sma200})
    
    # 信號：價格在 SMA200 之上為 1 (持有)，否則為 0 (空手)
    df['Signal'] = np.where(df['Close'] > df['SMA200'], 1, 0)
    
    # 計算策略淨值 (使用 shift(1) 避免前視偏誤)
    df['Strategy'] = (1 + df['Close'].pct_change() * df['Signal'].shift(1)).cumprod()
    df['BuyHold'] = (1 + df['Close'].pct_change()).cumprod()
    
    return df['Strategy'], df['BuyHold']

def parse_input(text):
    """
    解析使用者輸入的 CSV 格式持倉字串。
    """
    port = {}
    for line in text.strip().split('\n'):
        if ',' in line:
            parts = line.split(',')
            try: 
                port[parts[0].strip().upper()] = float(parts[1].strip())
            except: 
                port[parts[0].strip().upper()] = 0.0
    return port


# ==========================================
# 5. 主應用程式入口 (Main Application)
# ==========================================

def main():
    # --- 側邊欄配置 ---
    with st.sidebar:
        st.header("⚙️ 指揮系統設定")
        fred_key = st.secrets.get("FRED_API_KEY", st.text_input("FRED API Key", type="password"))
        
        # 資產配置輸入
        user_cash = st.number_input("💰 現金儲備 (USD)", value=10000.0, step=1000.0)
        user_input = st.text_area("持倉市值清單 (Ticker, Value)", "BTC-USD, 10000\nAMD, 10000\nNVDA, 10000", height=150)
        
        # 解析持倉
        p_dict = parse_input(user_input)
        tickers_list = list(p_dict.keys())
        total_assets = user_cash + sum(p_dict.values())
        
        st.metric("🏦 總資產", f"${total_assets:,.0f}", f"現金: ${user_cash:,.0f}")
        
        slot_limit = st.slider("預算上限 (%)", 5, 50, 20) / 100
        
        # 啟動按鈕
        if st.button("🚀 啟動 Alpha 指揮中心", type="primary"): 
            st.session_state['run'] = True

    # --- 主畫面邏輯 ---
    if not st.session_state.get('run', False):
        st.info("請於左側輸入資料並點擊【啟動 Alpha 指揮中心】以載入 Q1 2026 戰情。")
        return

    with st.spinner("🦅 Alpha 13.8 正在執行拓撲全域掃描 (含宏觀流動性修正)..."):
        # 獲取所有數據
        df_close, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_macro, df_fed = fetch_fred_macro(fred_key) # 可能回傳 None
        adv_data = {t: get_advanced_info(t) for t in tickers_list}

    # 建立分頁系統
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "🦅 戰略戰情", 
        "🐋 凱利與籌碼", 
        "🔍 個股體檢", 
        "🚦 策略回測", 
        "💰 CFO 財報", 
        "🏠 房貸目標", 
        "📊 實驗室"
    ])

    # === TAB 1: 宏觀與戰略指揮 ===
    with t1:
        st.title("🦅 Alpha 13.8: 混合戰略指揮中心")
        st.subheader("1. 宏觀戰情 (Tripwires Monitor)")
        
        # 計算宏觀變化量
        # Net Liquidity
        if df_macro is not None:
            liq_now = df_macro['Net_Liquidity'].iloc[-1]
            try:
                liq_prev = df_macro['Net_Liquidity'].iloc[-2]
                liq_chg = liq_now - liq_prev
            except:
                liq_chg = 0
            st.metric("💧 淨流動性", f"${liq_now:.2f}T", f"{liq_chg:+.2f}T")
        else:
            st.warning("⚠️ 無法獲取 FRED 數據，宏觀指標與回測修正將不可用。")
            liq_now = 0; liq_chg = 0

        # VIX
        vix_now = df_close['^VIX'].iloc[-1] if '^VIX' in df_close.columns else 0
        vix_prev = df_close['^VIX'].iloc[-2] if '^VIX' in df_close.columns and len(df_close) > 1 else vix_now
        vix_chg = vix_now - vix_prev

        # TNX (10Y Bond)
        tnx_now = df_close['^TNX'].iloc[-1] if '^TNX' in df_close.columns else 0
        tnx_prev = df_close['^TNX'].iloc[-2] if '^TNX' in df_close.columns and len(df_close) > 1 else tnx_now
        tnx_chg = tnx_now - tnx_prev

        # Copper/Gold
        try: 
            cg_series = (df_close['HG=F'] / df_close['GC=F']) * 1000
            cg_now = cg_series.iloc[-1]
            cg_prev = cg_series.iloc[-2] if len(cg_series) > 1 else cg_now
            cg_chg = cg_now - cg_prev
        except: 
            cg_now = 0; cg_chg = 0
        
        # Fed Rate
        fed_now = df_fed['Fed_Rate'].iloc[-1] if df_fed is not None else 0
        fed_prev = df_fed['Fed_Rate'].iloc[-2] if df_fed is not None and len(df_fed) > 1 else fed_now
        fed_chg = fed_now - fed_prev
        
        # 顯示宏觀儀表板 (Metric Columns)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌪️ VIX", f"{vix_now:.2f}", f"{vix_chg:+.2f}", delta_color="inverse")
        c2.metric("⚖️ 10年債", f"{tnx_now:.2f}%", f"{tnx_chg:+.2f}%", delta_color="inverse")
        c3.metric("🏭 銅金比", f"{cg_now:.2f}", f"{cg_chg:+.2f}")
        c4.metric("🏦 Fed利率", f"{fed_now:.2f}%", f"{fed_chg:+.2f}%", delta_color="inverse")
        
        st.markdown("#### 📊 CFO 混合戰略總表 (含 $±2\sigma$ 預測範圍)")
        summary = []
        for t in tickers_list:
            if t not in df_close.columns: continue
            
            # 獲取各項指標
            tr = analyze_trend_multi(df_close[t])
            targets = calc_targets_composite(t, df_close, df_high, df_low, adv_data.get(t,{}), 30)
            tgt = targets['Avg'] if targets else 0
            
            # 計算波動區間
            vol_22 = df_close[t].pct_change().std() * np.sqrt(22)
            pred_range = f"${tr['p_now']*(1-2*vol_22):.2f} - ${tr['p_now']*(1+2*vol_22):.2f}"
            
            rsi, slope, vol_r = calc_tech_indicators(df_close[t], df_vol[t])
            mvrv_z = calc_mvrv_z(df_close[t]).iloc[-1] if calc_mvrv_z(df_close[t]) is not None else 0
            six_s = calc_six_dim_state(df_close[t])
            
            # [關鍵修復] 呼叫 V2 回測，正確傳入 df_macro
            bt_res = run_backtest_lab_v2(t, df_close, df_high, df_low, df_macro, adv_data.get(t,{}), 30)
            
            # 獲取 CFO 指令
            cfo_act, b_pct = get_cfo_directive_v4(tr['p_now'], six_s, tr['status'], tr['is_bull'], rsi, slope, vol_r, mvrv_z, tgt*1.05, tgt*0.95)
            mode_tag = f'<span class="bull-mode">BULL</span>' if tr['is_bull'] else f'<span class="bear-mode">BEAR</span>'
            
            summary.append({
                "代號": t, 
                "模式": mode_tag, 
                "現價": f"${tr['p_now']:.2f}", 
                "CFO 指令": cfo_act, 
                "預期範圍(±2σ)": pred_range, 
                "目標價(Avg)": f"${tgt:.2f}",
                "回測誤差": f"{bt_res['Error']:.1%}" if bt_res else "N/A"
            })
            
        st.write(pd.DataFrame(summary).to_html(escape=False), unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("2. 個股雷達 (預測回測: 30天前 - 含宏觀修正)")
        for t in tickers_list:
            if t not in df_close.columns: continue
            
            # [關鍵修復] 呼叫 V2 回測
            bt_res = run_backtest_lab_v2(t, df_close, df_high, df_low, df_macro, adv_data.get(t,{}), 30)
            obv = calc_obv(df_close[t], df_vol[t])
            comp_res = compare_with_leverage(t, df_close)
            targets = calc_targets_composite(t, df_close, df_high, df_low, adv_data.get(t,{}), 30)
            
            with st.expander(f"🦅 {t} 戰略深度分析", expanded=False):
                k1, k2, k3 = st.columns([2, 1, 1])
                with k1: 
                    if comp_res: st.plotly_chart(px.line(comp_res[0], title=f"{t} vs TQQQ").update_layout(height=300), use_container_width=True)
                with k2:
                    st.markdown("#### 🎯 估值體系 (1M)")
                    if targets:
                        for key, val in targets.items(): st.write(f"**{key}:** ${val:.2f}" if val else f"**{key}:** N/A")
                    st.markdown("#### 🔄 拓撲回測驗證")
                    if bt_res and bt_res['Past_Pred']:
                        st.code(bt_res['Process'], language="text")
                    else: st.info("數據不足")
                with k3:
                    st.markdown("#### 🐋 籌碼與數據")
                    st.write(f"機構持股: {(adv_data.get(t,{}).get('Inst_Held') or 0)*100:.1f}%")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=df_close[t].iloc[-126:], name='Price'))
                    if obv is not None: 
                        fig.add_trace(go.Scatter(y=obv.iloc[-126:], name='OBV', yaxis='y2'))
                    fig.update_layout(height=300, yaxis2=dict(overlaying='y', side='right'))
                    st.plotly_chart(fig, use_container_width=True)

    # === TAB 2: 籌碼 & Kelly (新增動態凱利與主力動向) ===
    with t2:
        st.subheader("🐋 深度籌碼與動態凱利 (Dynamic Kelly)")
        st.info("ℹ️ 凱利公式參數: 歷史窗口 (Lookback) = 60天 (對應 2026 Q1 週期)。採用 Half-Kelly 策略。")
        chip_data = []
        for t in tickers_list:
            if t not in df_close.columns: continue
            
            # 呼叫更新後的動態凱利函數 (t=60)
            k_pct = calc_dynamic_kelly(df_close[t], lookback=60)
            obv_trend = calc_obv_trend(df_close[t], df_vol[t])
            info = adv_data.get(t, {})
            
            chip_data.append({
                "代號": t, 
                "主力動向 (OBV)": obv_trend,
                "機構持股": f"{(info.get('Inst_Held') or 0)*100:.1f}%", 
                "空單比率": f"{(info.get('Short_Ratio') or 0):.2f}",
                "動態凱利建議 (60d)": f"{k_pct*100:.1f}%"
            })
        st.dataframe(pd.DataFrame(chip_data), use_container_width=True)

    # === TAB 3: 體質 ===
    with t3:
        st.subheader("🔍 財務體質")
        h_data = [{
            "代號": t, 
            "PEG": f"{(adv_data.get(t,{}).get('PEG') or 0):.2f}", 
            "ROE": f"{(adv_data.get(t,{}).get('ROE') or 0)*100:.1f}%", 
            "淨利率": f"{(adv_data.get(t,{}).get('Profit_Margin') or 0)*100:.1f}%"
        } for t in tickers_list]
        st.dataframe(pd.DataFrame(h_data), use_container_width=True)

    # === TAB 4: SMA200 回測 ===
    with t4:
        st.subheader("🚦 SMA200 長期策略回測")
        for t in tickers_list:
            if t in df_close.columns:
                s, b = run_traffic_light(df_close[t])
                st.write(f"**{t}**")
                st.line_chart(pd.DataFrame({"策略": s, "買入持有": b}).dropna())

    # === TAB 5: CFO 財報 (Trigger-on-Click & No Limit) ===
    with t5:
        st.subheader("💰 CFO 財報與 Coast FIRE 模擬")
        st.write("請輸入參數後，點擊下方 **「執行財務健檢確認」** 按鈕以進行計算。")
        
        c1, c2, c3, c4 = st.columns(4)
        # 輸入區：無下限設定 (min_value=None)
        age = c1.number_input("目前年齡", value=35)
        r_age = c2.number_input("退休年齡", value=60)
        # 允許負資產或任意數值輸入
        net_worth = c3.number_input("目前淨資產 (TWD/USD)", value=2000000.0, step=100000.0, min_value=None)
        save = c4.number_input("每月儲蓄", value=30000.0)
        exp_ret = c1.number_input("預期年化報酬 (%)", value=7.0)
        infl = c2.number_input("通膨率 (%)", value=2.0)
        
        # 手動觸發按鈕
        if st.button("🧮 執行財務健檢確認", type="primary"):
            df_fire, final_bal = calc_coast_fire(age, r_age, net_worth, save, exp_ret, infl)
            
            k1, k2 = st.columns(2)
            k1.metric("退休時預估資產 (終值)", f"${final_bal:,.0f}")
            k2.metric("財務自由數字 (年支出的25倍)", f"${(save*12*25):,.0f} (估)")
            
            st.line_chart(df_fire.set_index("Age")['Balance'])
        else:
            st.warning("等待輸入確認...")

    # === TAB 6: 房貸目標 (Trigger-on-Click & No Limit) ===
    with t6:
        st.subheader("🏠 房貸目標 (提前還款模擬)")
        st.write("請輸入參數後，點擊下方 **「執行房貸分析」** 按鈕以進行計算。")
        
        c1, c2, c3 = st.columns(3)
        # 輸入區：無下限設定 (min_value=None)
        amt = c1.number_input("貸款總額 (無下限)", value=10000000.0, step=100000.0, min_value=None)
        rt = c2.number_input("年利率 (%)", value=2.2)
        yrs = c3.number_input("貸款年限", value=30)
        extra = st.number_input("每月額外還款 (Extra)", value=5000.0)
        
        # 手動觸發按鈕
        if st.button("🏠 執行房貸分析", type="primary"):
            pmt, tot_int_norm, tot_int_acc, saved_int, years_saved = calc_mortgage_advanced(amt, rt, yrs, extra)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("表定月付", f"${pmt:,.0f}")
            m2.metric("總利息節省", f"${saved_int:,.0f}", f"提早 {years_saved:.1f} 年還完")
            m3.metric("實際總利息", f"${tot_int_acc:,.0f}")
        else:
            st.warning("等待輸入確認...")

    # === TAB 7: 策略實驗室 (Topological Defensive Backtest) ===
    with t7:
        st.subheader("📊 拓撲實驗室 (宏觀防禦版)")
        st.write("此回測將引入 FRED 流動性因子：當 Fed 縮表時 (Net Liquidity 趨勢 < 0)，強制進入防禦模式。")
        
        lab_ticker = st.selectbox("選擇回測標的", sorted(list(set(tickers_list + ['TQQQ', 'QQQ', 'SPY']))))
        
        if lab_ticker in df_close.columns:
            # [關鍵修復] 執行 V3 回測，正確傳入 df_macro
            res, r_dca, r_strat, inv = run_strategy_backtest_salary_flow_v3(
                df_close[lab_ticker].to_frame(name='Close'), 
                df_vol[lab_ticker],
                df_macro # 正確傳遞宏觀數據
            )
            
            c1, c2, c3 = st.columns(3)
            c1.metric("投入本金", f"${inv:,.0f}")
            c2.metric("DCA 報酬率", f"{r_dca:.1%}")
            c3.metric("拓撲策略 報酬率", f"{r_strat:.1%}", delta=f"{(r_strat-r_dca)*100:.1f} pts")
            
            st.plotly_chart(px.line(res[['DCA', 'Strat']], title="淨值對比"), use_container_width=True)

if __name__ == "__main__":
    main()