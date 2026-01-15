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
st.set_page_config(page_title="Alpha 11.1: 動態凱利版", layout="wide", page_icon="🦅")

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
    benchmarks = ['SPY', 'QQQ', 'QLD', 'TQQQ', '^VIX', '^TNX', 'HYG', 'GC=F', 'HG=F', 'DX-Y.NYB'] 
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
        df = pd.DataFrame({'WALCL': walcl, 'TGA': tga, 'RRP': rrp}).ffill().dropna()
        df['Net_Liquidity'] = (df['WALCL'] - df['TGA'] - df['RRP']) / 1000 
        return df
    except: return None

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

# --- 2. 戰略運算 (AI & Logic) ---

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

def calc_six_dim_state(series):
    if len(series) < 22: return "N/A", "觀察"
    p = series.iloc[-1]
    ma20 = series.rolling(20).mean().iloc[-1]
    std20 = series.rolling(20).std().iloc[-1]
    up = ma20 + 2 * std20
    lw = ma20 - 2 * std20
    
    if p > up * 1.05: return "H3 極限噴出", "賣出 1/2 (清空槓桿)"
    if p > up: return "H2 情緒過熱", "賣出 1/3 (獲利了結)"
    if p > ma20: return "H1 多頭回歸", "續抱"
    if p < lw * 0.95: return "L3 恐慌崩盤", "分批接刀"
    if p < lw: return "L2 超賣區", "觀察買點"
    return "L1 震盪整理", "觀望"

def get_cfo_directive(p_now, six_state, trend_status, range_high):
    action = "🟩 續抱/觀察"
    if "L" in six_state and "空頭" in trend_status: action = "⬛ 止損/清倉 (趨勢損毀)"
    elif range_high > 0 and p_now >= range_high: action = "🟥 賣出 50% (達預測高點)"
    elif "H3" in six_state: action = "🟥 賣出 50% (極限噴出)"
    elif "H2" in six_state: action = "🟧 賣出 1/3 (過熱)"
    return action

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

# [NEW] 動態凱利公式 (Dynamic Kelly)
def calc_dynamic_kelly(series, lookback=63):
    """
    基於過去 3 個月(63天)的日回報計算凱利公式
    f* = W - (1-W)/R
    W = 勝率, R = 盈虧比
    回傳值已做 Half-Kelly 安全處理 (0~100%)
    """
    if len(series) < lookback: return 0.0
    
    # 取最近 N 天
    recent = series.iloc[-lookback:]
    rets = recent.pct_change().dropna()
    
    if len(rets) < 10: return 0.0
    
    # 區分勝負
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    
    # 極端情況防呆
    if len(losses) == 0: return 1.0 # 全勝 (罕見)
    if len(wins) == 0: return 0.0   # 全敗
    
    win_rate = len(wins) / len(rets)
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    
    if avg_loss == 0: return 1.0
    
    wl_ratio = avg_win / avg_loss # 盈虧比 R
    
    # Kelly Formula
    kelly = win_rate - ((1 - win_rate) / wl_ratio)
    
    # Half-Kelly for Safety (且限制在 0 ~ 100% 之間)
    kelly_safe = max(0.0, min(1.0, kelly * 0.5))
    
    return kelly_safe

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
        if st.button("🚀 啟動凱利版", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False): return

    with st.spinner("🦅 Alpha 11.1 正在計算最佳持倉比例..."):
        df_close, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_macro = fetch_fred_macro(fred_key)
        adv_data = {t: get_advanced_info(t) for t in tickers_list}

    if df_close.empty: st.error("❌ 無數據"); st.stop()

    t1, t2, t3, t4, t5, t6 = st.tabs(["🦅 戰略戰情", "🐋 深度籌碼", "🔍 個股體檢", "🚦 策略回測", "💰 CFO 財報", "🏠 房貸目標"])

    # === TAB 1: 戰略 ===
    with t1:
        st.subheader("1. 宏觀與總表")
        liq = df_macro['Net_Liquidity'].iloc[-1] if df_macro is not None else 0
        vix = df_close['^VIX'].iloc[-1] if '^VIX' in df_close.columns else 0
        tnx = df_close['^TNX'].iloc[-1] if '^TNX' in df_close.columns else 0
        try: cg = (df_close['HG=F'].iloc[-1]/df_close['GC=F'].iloc[-1])*1000
        except: cg = 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💧 淨流動性", f"${liq:.2f}T")
        c2.metric("🌪️ VIX", f"{vix:.2f}", delta_color="inverse")
        c3.metric("⚖️ 10年殖利率", f"{tnx:.2f}%")
        c4.metric("🏭 銅金比", f"{cg:.2f}")

        if df_macro is not None: st.plotly_chart(px.line(df_macro, y='Net_Liquidity', title='聯準會流動性趨勢', height=250), use_container_width=True)

        st.markdown("#### 📊 CFO 戰略指令總表 (含動態凱利)")
        summary = []
        for t in tickers_list:
            if t not in df_close.columns: continue
            
            # 基本運算
            trend = analyze_trend_multi(df_close[t])
            targets = calc_targets_composite(t, df_close, df_high, df_low, adv_data.get(t,{}), 22)
            bt = run_backtest_lab(t, df_close, df_high, df_low, 22)
            
            # 六維狀態
            six_state, six_desc = calc_six_dim_state(df_close[t])
            
            # 動態 Kelly 計算 (近3個月)
            d_kelly = calc_dynamic_kelly(df_close[t], lookback=63)
            
            # 信心區間
            vol_daily = df_close[t].pct_change().std()
            price_sigma = df_close[t].iloc[-1] * vol_daily * np.sqrt(22)
            tgt_val = targets['Avg'] if targets and targets['Avg'] else 0
            
            range_high = 0; range_str = "-"
            if tgt_val > 0:
                range_low = tgt_val - 2 * price_sigma
                range_high = tgt_val + 2 * price_sigma
                range_str = f"${range_low:.0f} ~ ${range_high:.0f}"
            
            cfo_act = get_cfo_directive(trend['p_now'], six_state, trend['status'], range_high)
            
            summary.append({
                "代號": t, 
                "現價": f"${trend['p_now']:.2f}", 
                "CFO 指令": cfo_act,
                "動態 Kelly (3M)": f"{d_kelly*100:.1f}%",
                "預測值 (1M)": f"${tgt_val:.2f}" if tgt_val > 0 else "-",
                "95% 區間": range_str,
                "回測 Bias": f"{bt['Error']:.1%}" if bt else "-"
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
        st.caption("📝 動態 Kelly：基於過去 3 個月勝率與盈虧比計算的最佳倉位 (已啟用 Half-Kelly 安全模式)。")
        
        st.markdown("---")
        st.subheader("2. 個股戰略雷達")
        
        for t in tickers_list:
            if t not in df_close.columns: continue
            targets = calc_targets_composite(t, df_close, df_high, df_low, adv_data.get(t,{}), 22)
            bt = run_backtest_lab(t, df_close, df_high, df_low, 22)
            obv = calc_obv(df_close[t], df_vol[t])
            mvrv_s = calc_mvrv_z(df_close[t])
            mvrv = mvrv_s.iloc[-1] if mvrv_s is not None else 0
            comp_res = compare_with_leverage(t, df_close)
            six_state, _ = calc_six_dim_state(df_close[t])
            d_kelly = calc_dynamic_kelly(df_close[t], 63)
            
            t_avg = f"${targets['Avg']:.2f}" if targets and targets['Avg'] else "-"
            
            with st.expander(f"🦅 {t} | Kelly: {d_kelly*100:.0f}% | {six_state}", expanded=False):
                k1, k2, k3 = st.columns([2, 1, 1])
                with k1:
                    if comp_res:
                        st.markdown(f"**槓桿挑戰:** {comp_res[1]}")
                        st.plotly_chart(px.line(comp_res[0], title=f"{t} vs TQQQ").update_layout(height=300), use_container_width=True)
                    else: st.write("無數據")

                with k2:
                    st.markdown("#### 🎯 五角定位 (1M)")
                    if targets:
                        st.write(f"**ATR (物理):** ${targets['ATR']:.2f}" if targets['ATR'] else "-")
                        st.write(f"**MC (機率):** ${targets['MC']:.2f}" if targets['MC'] else "-")
                        st.write(f"**Fib (心理):** ${targets['Fib']:.2f}" if targets['Fib'] else "-")
                        st.write(f"**RF (AI):** ${targets['RF']:.2f}" if targets['RF'] else "-")
                        st.write(f"**Fund (DCF):** ${targets['Fund']}" if targets['Fund'] else "N/A")
                    if bt:
                        st.markdown(f"回測誤差: **{bt['Error']:.1%}**")

                with k3:
                    st.markdown("#### 📉 雙軸資金流")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_close.index[-126:], y=df_close[t].iloc[-126:], name='Price', line=dict(color='#00FF7F')))
                    if obv is not None:
                        fig.add_trace(go.Scatter(x=df_close.index[-126:], y=obv.iloc[-126:], name='OBV', line=dict(color='#FFD700', width=1), yaxis='y2'))
                    fig.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0), yaxis2=dict(overlaying='y', side='right', showgrid=False))
                    st.plotly_chart(fig, use_container_width=True)

    # === TAB 2: 籌碼 ===
    with t2:
        st.subheader("🐋 籌碼與內部人")
        chip_data = []
        for t in tickers_list:
            if t not in df_close.columns: continue
            info = adv_data.get(t, {})
            inst = info.get('Inst_Held')
            insider = info.get('Insider_Held')
            short = info.get('Short_Ratio')
            chip_data.append({
                "代號": t,
                "機構持股": f"{inst*100:.1f}%" if inst is not None else "-",
                "內部人持股": f"{insider*100:.1f}%" if insider is not None else "-",
                "空單比例": f"{short:.2f}" if short is not None else "-"
            })
        st.dataframe(pd.DataFrame(chip_data), use_container_width=True)

    # === TAB 3: 體質 ===
    with t3:
        st.subheader("🔍 財務體質掃描")
        health_data = []
        for t in tickers_list:
            info = adv_data.get(t, {})
            is_etf = info.get('Type') == 'ETF'
            peg = info.get('PEG'); peg_s = "ETF" if is_etf else (f"{peg:.2f}" if peg is not None else "-")
            roe = info.get('ROE'); roe_s = "ETF" if is_etf else (f"{roe*100:.1f}%" if roe is not None else "-")
            pm = info.get('Profit_Margin'); pm_s = "ETF" if is_etf else (f"{pm*100:.1f}%" if pm is not None else "-")
            
            health_data.append({
                "代號": t, "PEG": peg_s, "ROE": roe_s, "淨利率": pm_s,
                "流動比": info.get('Current_Ratio'), "負債/權益": info.get('Debt_Equity')
            })
        st.dataframe(pd.DataFrame(health_data), use_container_width=True)

    # === TAB 4~6 (保留) ===
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

if __name__ == "__main__":
    main()