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
    page_title="Alpha 13.9: 拓撲指揮官 (Final)",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# 注入 CSS 樣式
st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #444; border-radius: 5px; padding: 15px; color: white;}
    .bull-mode {color: #00FF7F; font-weight: bold; border: 1px solid #00FF7F; padding: 2px 8px; border-radius: 4px; font-size: 0.9em;}
    .bear-mode {color: #FF4B4B; font-weight: bold; border: 1px solid #FF4B4B; padding: 2px 8px; border-radius: 4px; font-size: 0.9em;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {height: 50px; background-color: #1E1E1E; border-radius: 5px 5px 0 0; color: white;}
    .stTabs [aria-selected="true"] {background-color: #00BFFF; color: black;}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 1. 核心數據引擎 (Data Sheaf Engine)
# ==========================================

@st.cache_data(ttl=1800)
def fetch_market_data(tickers):
    """
    獲取市場價格數據，構建基礎單純複形。
    """
    benchmarks = ['SPY', 'QQQ', 'QLD', 'TQQQ', '^VIX', '^TNX', '^IRX', 'HYG', 'GC=F', 'HG=F', 'DX-Y.NYB'] 
    all_tickers = list(set(tickers + benchmarks))
    data = {col: {} for col in ['Close', 'Open', 'High', 'Low', 'Volume']}
    for t in all_tickers:
        try:
            df = yf.Ticker(t).history(period="2y", auto_adjust=True)
            if df.empty: continue
            data['Close'][t] = df['Close']; data['Open'][t] = df['Open']
            data['High'][t] = df['High']; data['Low'][t] = df['Low']; data['Volume'][t] = df['Volume']
        except Exception: continue
    return pd.DataFrame(data['Close']).ffill(), pd.DataFrame(data['High']).ffill(), pd.DataFrame(data['Low']).ffill(), pd.DataFrame(data['Volume']).ffill()

@st.cache_data(ttl=3600*12)
def fetch_fred_macro(api_key):
    """
    獲取宏觀數據 (Global Section)。
    """
    if not api_key: return None, None
    try:
        fred = Fred(api_key=api_key)
        walcl = fred.get_series('WALCL', observation_start='2024-01-01')
        tga = fred.get_series('WTREGEN', observation_start='2024-01-01')
        rrp = fred.get_series('RRPONTSYD', observation_start='2024-01-01')
        fed_rate = fred.get_series('FEDFUNDS', observation_start='2023-01-01')
        
        df = pd.DataFrame({'WALCL': walcl, 'TGA': tga, 'RRP': rrp}).ffill().dropna()
        # 單位轉換為 Trillion (兆美元)
        df['Net_Liquidity'] = (df['WALCL'] - df['TGA'] - df['RRP']) / 1000 
        df_rate = pd.DataFrame({'Fed_Rate': fed_rate}).resample('D').ffill()
        return df, df_rate
    except Exception: return None, None

@st.cache_data(ttl=3600*24)
def get_advanced_info(ticker):
    try:
        t = yf.Ticker(ticker); info = t.info
        return {
            'Type': 'ETF' if 'ETF' in info.get('quoteType', '').upper() else 'Stock',
            'Target_Mean': info.get('targetMeanPrice'), 
            'PEG': info.get('pegRatio'),
            'Inst_Held': info.get('heldPercentInstitutions'), 
            'Short_Ratio': info.get('shortRatio'), 
            'ROE': info.get('returnOnEquity'),
            'Profit_Margin': info.get('profitMargins'),
            'Sector': info.get('sector', 'Unknown') # 用於分類防禦
        }
    except Exception: return {}

# ==========================================
# 2. 戰略模型 (Strategic Algorithms)
# ==========================================

def train_rf_model(df_close, ticker, days_forecast=30):
    try:
        if ticker not in df_close.columns: return None
        df = pd.DataFrame({'Close': df_close[ticker]})
        df['Ret'] = df['Close'].pct_change()
        df['Vol'] = df['Ret'].rolling(20).std()
        df['Target'] = df['Close'].shift(-days_forecast)
        df = df.dropna()
        if len(df) < 60: return None
        
        X = df[['Ret', 'Vol']]
        y = df['Target']
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X, y)
        return model.predict(X.iloc[[-1]])[0]
    except: return None

def calc_targets_composite(ticker, df_close, df_high, df_low, f_data, days_forecast=30):
    if ticker not in df_close.columns: return None
    c = df_close[ticker]; h = df_high[ticker]; l = df_low[ticker]
    try:
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        t_atr = c.iloc[-1] + (tr.rolling(14).mean().iloc[-1] * np.sqrt(days_forecast))
        
        mu = c.pct_change().mean()
        t_mc = c.iloc[-1] * ((1 + mu)**days_forecast)
        
        recent = c.iloc[-60:]
        t_fib = recent.max() + (recent.max() - recent.min()) * 0.618 
        t_rf = train_rf_model(df_close, ticker, days_forecast)
        t_fund = f_data.get('Target_Mean')

        targets = [t for t in [t_atr, t_mc, t_fib, t_fund, t_rf] if t is not None and not pd.isna(t)]
        t_avg = sum(targets)/len(targets) if targets else None
        return {"Avg": t_avg, "ATR": t_atr, "MC": t_mc, "Fib": t_fib, "RF": t_rf}
    except: return None

# --- [V2.1] 拓撲雷達回測 (注入實驗參數 -0.137T) ---
def run_backtest_lab_v2(ticker, df_close, df_high, df_low, df_macro, f_data, days_ago=30):
    if ticker not in df_close.columns or len(df_close) < 250: return None
    
    idx_past = len(df_close) - days_ago - 1
    date_past = df_close.index[idx_past]
    p_past = df_close[ticker].iloc[idx_past]
    p_now = df_close[ticker].iloc[-1]
    
    # 宏觀狀態檢查
    macro_status = "⚪ 中性"
    is_contraction = False
    
    if df_macro is not None and not df_macro.empty:
        try:
            # 找到最接近的日期
            m_idx = df_macro.index.get_indexer([date_past], method='ffill')[0]
            if m_idx > 20:
                liq_curr = df_macro['Net_Liquidity'].iloc[m_idx]
                liq_prev = df_macro['Net_Liquidity'].iloc[m_idx - 20]
                
                # [實驗參數更新] 最佳閾值 -0.137T
                if (liq_curr - liq_prev) < -0.137: 
                    is_contraction = True
                    macro_status = "🔻 緊縮 (Risk-Off)"
                elif (liq_curr - liq_prev) > 0.05:
                    macro_status = "💧 寬鬆 (Risk-On)"
        except: pass

    # 過去的預測
    df_p = df_close.iloc[:idx_past+1]; h_p = df_high.iloc[:idx_past+1]; l_p = df_low.iloc[:idx_past+1]
    raw_targets = calc_targets_composite(ticker, df_p, h_p, l_p, f_data, days_ago)
    final_pred = raw_targets['Avg'] if raw_targets else None
    
    # 拓撲修正 (Topological Correction)
    note = ""
    if is_contraction and final_pred and final_pred > p_past:
        # 實驗證明：緊縮期預測誤差大，需大幅下修
        final_pred = final_pred * 0.85 
        note = "(觸發拓撲盾牌)"

    if final_pred:
        err = (final_pred - p_now) / p_now
        process = f"[{macro_status}] 預測: {final_pred:.2f} {note} vs 現價: {p_now:.2f} | 誤差: {err:.1%}"
    else:
        err = 0; process = "N/A"
        
    return {"Past_Pred": final_pred, "Present_Value": p_now, "Error": err, "Process": process}

def analyze_trend_multi(series):
    if len(series) < 200: return {"status": "資料不足", "p_now": series.iloc[-1], "is_bull": False}
    p = series.iloc[-1]; sma200 = series.rolling(200).mean().iloc[-1]
    sma200_prev = series.rolling(200).mean().iloc[-10]
    is_bull = (p > sma200) and (sma200 > sma200_prev)
    return {"status": "🔥 多頭" if p > sma200 else "🛑 空頭", "p_now": p, "is_bull": is_bull}

def calc_dynamic_kelly(series, lookback=60):
    try:
        rets = series.iloc[-lookback:].pct_change().dropna()
        wins = rets[rets > 0]; losses = rets[rets < 0]
        if len(losses) == 0: return 0.5
        win_rate = len(wins) / len(rets)
        avg_win = wins.mean(); avg_loss = abs(losses.mean())
        if avg_loss == 0: return 0.5
        kelly = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        return max(0.0, min(1.0, kelly * 0.5))
    except: return 0.0

def calc_tech_indicators(series, vol_series):
    if len(series) < 60: return 50, 0, 1
    delta = series.diff()
    up = delta.clip(lower=0); down = -1 * delta.clip(upper=0)
    rs = up.ewm(com=13).mean() / down.ewm(com=13).mean()
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    ma20 = series.rolling(20).mean()
    slope = (ma20.iloc[-1] - ma20.iloc[-5]) / ma20.iloc[-5]
    vol_ma = vol_series.rolling(20).mean().iloc[-1]
    vol_ratio = vol_series.iloc[-1] / vol_ma if vol_ma > 0 else 1.0
    return rsi, slope, vol_ratio

def calc_six_dim_state(series):
    if len(series) < 22: return "N/A"
    p = series.iloc[-1]; ma20 = series.rolling(20).mean().iloc[-1]; std = series.rolling(20).std().iloc[-1]
    if p > ma20 + 2*std * 1.05: return "H3 極限噴出"
    if p > ma20 + 2*std: return "H2 情緒過熱"
    if p > ma20: return "H1 多頭回歸"
    if p < ma20 - 2*std: return "L2 超賣區"
    return "L1 震盪整理"

def get_cfo_directive_v4(p_now, six_state, trend_status, bull_mode, rsi, slope, vol_ratio, mvrv_z, range_high, range_low):
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

def calc_obv_trend(close, volume, lookback=20):
    try:
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        if len(obv) < lookback: return "N/A"
        delta = obv.iloc[-1] - obv.iloc[-lookback]
        if delta > 0: return "🔥 吸籌 (買入)"
        else: return "🔻 出貨 (賣出)"
    except: return "N/A"

def calc_obv(close, volume):
    if volume is None: return None
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

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

# ==========================================
# 3. 財務深度計算
# ==========================================

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

# ==========================================
# 4. 回測實驗室 V3 (拓撲防禦版 - 增強)
# ==========================================

def run_strategy_backtest_salary_flow_v3(df_in, vol_in, df_macro, ticker_type="Growth"):
    """
    V3 回測 (Final): 
    - 引入實驗參數 -0.137T
    - 區分 ticker_type (Growth/Crypto vs Defensive/Stable)
    """
    df = df_in.copy(); df['Volume'] = vol_in
    if len(df) > 500: df = df.iloc[-500:]
    
    # 宏觀對齊
    if df_macro is not None and not df_macro.empty:
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
        
        # 發薪日 DCA
        if d.month != last_m:
            cash_d += 10000; cash_s += 10000; inv += 10000; last_m = d.month
            buy = cash_d // p; stock_d += buy; cash_d -= buy * p
            
        if i > 20:
            ma20 = df['SMA20'].iloc[i]; ma200 = df['SMA200'].iloc[i]; rsi = df['RSI'].iloc[i]
            
            # --- 拓撲決策 (Experiment Logic) ---
            # [參數更新] 最佳閾值 -0.137T
            is_crunch = liq_trend < -0.137
            
            # [分類防禦]
            # Crypto/Growth: 一觸發緊縮就進入 "HARD DEFENSE" (空手)
            # Defensive/Semi: 觸發緊縮進入 "SOFT DEFENSE" (只賣一半)
            if is_crunch:
                if ticker_type in ["Crypto", "Growth", "High_Beta"]:
                    risk_mode = "HARD_DEFENSE"
                else:
                    risk_mode = "SOFT_DEFENSE"
            else:
                risk_mode = "NORMAL"
            
            # 賣出邏輯
            sell = 0
            if risk_mode == "HARD_DEFENSE":
                sell = 1.0 # 誤差大，必須清倉
            elif risk_mode == "SOFT_DEFENSE":
                if p < ma20: sell = 0.5 # 誤差小，減碼即可
            else:
                # 正常模式
                if p < ma20 and p < ma200: sell = 1.0
                elif rsi > 80: sell = 0.5
                
            if sell > 0 and stock_s > 0:
                s_amt = int(stock_s * sell); stock_s -= s_amt; cash_s += s_amt * p
                
            # 買入邏輯
            if sell == 0:
                # 危機模式下禁止買入，除非是 Defensive
                can_buy = True
                if risk_mode == "HARD_DEFENSE": can_buy = False
                
                if can_buy:
                    bull = p > ma200
                    alloc = 0.8 if bull else 0.2
                    if cash_s > 100:
                        b_amt = cash_s * alloc // p; stock_s += b_amt; cash_s -= b_amt * p
                    
        hist.append({"Date": d, "DCA": cash_d + stock_d*p, "Strat": cash_s + stock_s*p})
        
    res = pd.DataFrame(hist).set_index("Date")
    # 安全除法
    final_d = (res['DCA'].iloc[-1]-inv)/inv if inv > 0 else 0
    final_s = (res['Strat'].iloc[-1]-inv)/inv if inv > 0 else 0
    return res, final_d, final_s, inv

def run_traffic_light(series):
    sma200 = series.rolling(200).mean()
    df = pd.DataFrame({'Close': series, 'SMA200': sma200})
    df['Signal'] = np.where(df['Close'] > df['SMA200'], 1, 0)
    df['Strategy'] = (1 + df['Close'].pct_change() * df['Signal'].shift(1)).cumprod()
    df['BuyHold'] = (1 + df['Close'].pct_change()).cumprod()
    return df['Strategy'], df['BuyHold']

def parse_input(text):
    port = {}
    for line in text.strip().split('\n'):
        if ',' in line:
            parts = line.split(',')
            try: port[parts[0].strip().upper()] = float(parts[1].strip())
            except: pass
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
        user_input = st.text_area("持倉市值清單 (Ticker, Value)", "BTC-USD, 10000\nNVDA, 10000", height=150)
        
        # 解析持倉
        p_dict = parse_input(user_input)
        tickers_list = list(p_dict.keys())
        total_assets = user_cash + sum(p_dict.values())
        
        st.metric("🏦 總資產", f"${total_assets:,.0f}", f"現金: ${user_cash:,.0f}")
        
        if st.button("🚀 啟動 Alpha 指揮中心", type="primary"): 
            st.session_state['run'] = True

    # --- 主畫面邏輯 ---
    if not st.session_state.get('run', False):
        st.info("請於左側輸入資料並點擊【啟動 Alpha 指揮中心】以載入 Q1 2026 戰情。")
        return

    with st.spinner("🦅 Alpha 13.9 正在執行拓撲全域掃描 (已套用實驗參數 -0.137T)..."):
        df_close, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_macro, df_fed = fetch_fred_macro(fred_key)
        adv_data = {t: get_advanced_info(t) for t in tickers_list}

    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "🦅 戰略戰情", "🐋 籌碼", "🔍 體質", "🚦 回測", "💰 CFO", "🏠 房貸", "📊 實驗室"
    ])

    # === TAB 1: 宏觀與戰略指揮 ===
    with t1:
        st.title("🦅 Alpha 13.9: 混合戰略指揮中心 (Final)")
        st.subheader("1. 宏觀戰情 (Tripwires Monitor)")
        
        if df_macro is not None:
            liq_now = df_macro['Net_Liquidity'].iloc[-1]
            try:
                liq_prev = df_macro['Net_Liquidity'].iloc[-2]
                liq_chg = liq_now - liq_prev
            except: liq_chg = 0
            
            # 顯示是否觸發 -0.137T 閾值
            is_crunch = (liq_now - df_macro['Net_Liquidity'].iloc[-20]) < -0.137 if len(df_macro)>20 else False
            status_text = "🚨 拓撲撕裂 (HARD DEFENSE)" if is_crunch else "✅ 流動性安全"
            
            st.metric("💧 淨流動性", f"${liq_now:.2f}T", f"{liq_chg:+.2f}T")
            if is_crunch: st.error(f"⚠️ 警告：20日流動性收縮觸發閾值 (-0.137T)！{status_text}")
        else:
            st.warning("⚠️ 無法獲取 FRED 數據，宏觀指標與回測修正將不可用。")
            liq_now = 0; liq_chg = 0

        # VIX, TNX, Etc.
        vix_now = df_close['^VIX'].iloc[-1] if '^VIX' in df_close.columns else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("🌪️ VIX", f"{vix_now:.2f}")
        
        st.markdown("#### 📊 CFO 混合戰略總表")
        summary = []
        for t in tickers_list:
            if t not in df_close.columns: continue
            tr = analyze_trend_multi(df_close[t])
            
            # V2 回測 (含 -0.137T 修正)
            bt_res = run_backtest_lab_v2(t, df_close, df_high, df_low, df_macro, adv_data.get(t,{}), 30)
            
            tgts = calc_targets_composite(t, df_close, df_high, df_low, adv_data.get(t,{}), 30)
            t_val = tgts['Avg'] if tgts else 0
            
            rsi, slope, vr = calc_tech_indicators(df_close[t], df_vol[t])
            six = calc_six_dim_state(df_close[t])
            act, _ = get_cfo_directive_v4(tr['p_now'], six, tr['status'], tr['is_bull'], rsi, slope, vr, 0, t_val*1.05, t_val*0.95)
            
            summary.append({
                "代號": t, "現價": f"${tr['p_now']:.2f}", 
                "CFO指令": act, "目標價": f"${t_val:.2f}",
                "回測誤差": f"{bt_res['Error']:.1%}" if bt_res else "N/A"
            })
        st.dataframe(pd.DataFrame(summary))
        
        st.markdown("---")
        st.write("🦅 **個股 30 天前預測驗證 (含實驗參數修正)**")
        for t in tickers_list:
            res = run_backtest_lab_v2(t, df_close, df_high, df_low, df_macro, adv_data.get(t,{}), 30)
            if res: st.text(f"{t}: {res['Process']}")

    # === TAB 2-6 (保持原樣，僅顯示關鍵功能) ===
    with t2:
        st.subheader("🐋 動態凱利籌碼")
        c_data = []
        for t in tickers_list:
            if t not in df_close.columns: continue
            k = calc_dynamic_kelly(df_close[t])
            c_data.append({"代號": t, "動態凱利%": f"{k*100:.1f}%"})
        st.dataframe(pd.DataFrame(c_data))
    
    with t3: st.dataframe(pd.DataFrame([{"代號": t, "ROE": adv_data.get(t,{}).get('ROE')} for t in tickers_list]))
    with t4: 
        for t in tickers_list: 
            if t in df_close.columns: st.line_chart(run_traffic_light(df_close[t])[0])

    with t5: # CFO
        st.subheader("CFO 財報")
        nw = st.number_input("淨資產", value=2000000.0, min_value=None)
        if st.button("計算FIRE"):
            df_f, bal = calc_coast_fire(35, 60, nw, 30000, 7, 2)
            st.metric("預估資產", f"${bal:,.0f}")
            st.line_chart(df_f.set_index("Age")['Balance'])

    with t6: # 房貸
        st.subheader("房貸試算")
        amt = st.number_input("貸款", value=10000000.0, min_value=None)
        if st.button("計算房貸"):
            pmt, _, tot_a, sav, _ = calc_mortgage_advanced(amt, 2.2, 30, 5000)
            st.metric("月付", f"${pmt:,.0f}")
            st.metric("省息", f"${sav:,.0f}")

    # === TAB 7: 策略實驗室 (V3 Final - 分類防禦) ===
    with t7:
        st.subheader("📊 拓撲實驗室 (V3 Final - 分類防禦)")
        st.write("引入實驗參數：閾值 -0.137T。Crypto/Growth 觸發時清倉，Defensive 觸發時減半。")
        
        lab_ticker = st.selectbox("選擇回測標的", sorted(list(set(tickers_list + ['TQQQ', 'QQQ', 'SPY']))))
        
        # 自動判斷類型 (簡單版)
        t_type = "Growth" # 預設
        if lab_ticker in ['BTC-USD', 'ETH-USD', 'ARKK', 'PLTR', 'NVDA', 'AMD']: t_type = "Growth"
        elif lab_ticker in ['KO', 'MCD', 'JNJ', 'PG', '2330.TW']: t_type = "Defensive"
        
        st.info(f"偵測到標的類型: {t_type}")
        
        if lab_ticker in df_close.columns:
            # 執行 V3 回測
            res, r_dca, r_strat, inv = run_strategy_backtest_salary_flow_v3(
                df_close[lab_ticker].to_frame(name='Close'), 
                df_vol[lab_ticker],
                df_macro,
                ticker_type=t_type # 傳入類型
            )
            
            c1, c2, c3 = st.columns(3)
            c1.metric("投入本金", f"${inv:,.0f}")
            c2.metric("DCA 報酬率", f"{r_dca:.1%}")
            c3.metric("拓撲策略 報酬率", f"{r_strat:.1%}", delta=f"{(r_strat-r_dca)*100:.1f} pts")
            
            st.plotly_chart(px.line(res[['DCA', 'Strat']], title=f"{lab_ticker} 淨值走勢"), use_container_width=True)

if __name__ == "__main__":
    main()