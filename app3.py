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
st.set_page_config(page_title="Alpha 12.6: 指揮官錢包", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #444; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
    .action-sell {color: #FF4B4B; font-weight: bold; background-color: #330000; padding: 2px 5px; border-radius: 3px;}
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
            data['Close'][t] = df['Close']; data['Open'][t] = df['Open']
            data['High'][t] = df['High']; data['Low'][t] = df['Low']; data['Volume'][t] = df['Volume']
        except: continue
    try:
        return (pd.DataFrame(data['Close']).ffill(), pd.DataFrame(data['High']).ffill(), 
                pd.DataFrame(data['Low']).ffill(), pd.DataFrame(data['Volume']).ffill())
    except: return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

@st.cache_data(ttl=3600*12)
def fetch_fred_macro(api_key):
    if not api_key: return None, None
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
        t = yf.Ticker(ticker); info = t.info; q_type = info.get('quoteType', '').upper()
        is_etf = 'ETF' in q_type or 'MUTUALFUND' in q_type
        peg = info.get('pegRatio'); fwd_pe = info.get('forwardPE'); earn_growth = info.get('earningsGrowth')
        if peg is None and fwd_pe is not None and earn_growth is not None and earn_growth > 0:
            peg = fwd_pe / (earn_growth * 100)
        rev_g = info.get('revenueGrowth'); pm = info.get('profitMargins')
        r40 = (rev_g + pm) * 100 if (rev_g is not None and pm is not None) else None
        return {'Type': 'ETF' if is_etf else 'Stock', 'Target_Mean': info.get('targetMeanPrice'), 'Forward_PE': fwd_pe, 'PEG': peg, 'Inst_Held': info.get('heldPercentInstitutions'), 'Insider_Held': info.get('heldPercentInsiders'), 'Short_Ratio': info.get('shortRatio'), 'Current_Ratio': info.get('currentRatio'), 'Debt_Equity': info.get('debtToEquity'), 'ROE': info.get('returnOnEquity'), 'Profit_Margin': pm, 'Rule_40': r40}
    except: return {'Type': 'Unknown'}

# --- 2. 戰略運算 ---

def train_rf_model(df_close, ticker, days_forecast=22):
    try:
        if ticker not in df_close.columns: return None
        df = pd.DataFrame(index=df_close.index); df['Close'] = df_close[ticker]
        df['Ret'] = df['Close'].pct_change(); df['Vol'] = df['Ret'].rolling(20).std(); df['SMA'] = df['Close'].rolling(20).mean()
        if '^VIX' in df_close.columns: df['VIX'] = df_close['^VIX']
        if '^TNX' in df_close.columns: df['TNX'] = df_close['^TNX']
        df['Target'] = df['Close'].shift(-days_forecast); df = df.dropna()
        if len(df) < 60: return None
        X = df.drop(columns=['Target', 'Close']); y = df['Target']
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42); model.fit(X, y)
        return model.predict(X.iloc[[-1]])[0]
    except: return None

def calc_targets_composite(ticker, df_close, df_high, df_low, f_data, days_forecast=22):
    if ticker not in df_close.columns: return None
    c = df_close[ticker]; h = df_high[ticker]; l = df_low[ticker]
    if len(c) < 100: return None
    try:
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1); atr = tr.rolling(14).mean().iloc[-1]; t_atr = c.iloc[-1] + (atr * np.sqrt(days_forecast))
    except: t_atr = None
    try:
        mu = c.pct_change().mean(); t_mc = c.iloc[-1] * ((1 + mu)**days_forecast)
    except: t_mc = None
    try:
        recent = c.iloc[-60:]; t_fib = recent.max() + (recent.max() - recent.min()) * 0.618 
    except: t_fib = None
    t_fund = f_data.get('Target_Mean'); t_rf = train_rf_model(df_close, ticker, days_forecast)
    targets = [t for t in [t_atr, t_mc, t_fib, t_fund, t_rf] if t is not None and not pd.isna(t)]; t_avg = sum(targets) / len(targets) if targets else None
    return {"ATR": t_atr, "MC": t_mc, "Fib": t_fib, "Fund": t_fund, "RF": t_rf, "Avg": t_avg}

def run_backtest_lab(ticker, df_close, df_high, df_low, days_ago=22):
    if ticker not in df_close.columns or len(df_close) < 250: return None
    idx_past = len(df_close) - days_ago - 1; p_now = df_close[ticker].iloc[-1]; df_past = df_close.iloc[:idx_past+1]
    past_rf = train_rf_model(df_past, ticker, days_ago); c_slice = df_close[ticker].iloc[:idx_past+1]; h_slice = df_high[ticker].iloc[:idx_past+1]; l_slice = df_low[ticker].iloc[:idx_past+1]
    tr = pd.concat([h_slice-l_slice], axis=1).max(axis=1); atr = tr.rolling(14).mean().iloc[-1]; past_atr = c_slice.iloc[-1] + (atr * np.sqrt(days_ago)); past_mc = c_slice.iloc[-1] * ((1 + c_slice.pct_change().mean())**days_ago)
    valid_past = [x for x in [past_rf, past_atr, past_mc] if x is not None]; past_avg = sum(valid_past) / len(valid_past) if valid_past else None
    err = (past_avg - p_now) / p_now if past_avg else 0
    return {"Past_Pred": past_avg, "Error": err}

def calc_mvrv_z(series):
    if len(series) < 200: return None
    sma200 = series.rolling(200).mean(); std200 = series.rolling(200).std(); return (series - sma200) / std200

def calc_tech_indicators(series, vol_series):
    if len(series) < 60: return None, None, None
    delta = series.diff(); up = delta.clip(lower=0); down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean(); ema_down = down.ewm(com=13, adjust=False).mean(); rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    ma20 = series.rolling(20).mean(); slope = (ma20.iloc[-1] - ma20.iloc[-5]) / ma20.iloc[-5]
    vol_ma = vol_series.rolling(20).mean().iloc[-1]; vol_ratio = vol_series.iloc[-1] / vol_ma if vol_ma > 0 else 1.0
    return rsi, slope, vol_ratio

def calc_six_dim_state(series):
    if len(series) < 22: return "N/A"
    p = series.iloc[-1]; ma20 = series.rolling(20).mean().iloc[-1]; std20 = series.rolling(20).std().iloc[-1]; up = ma20 + 2 * std20; lw = ma20 - 2 * std20
    if p > up * 1.05: return "H3 極限噴出"
    if p > up: return "H2 情緒過熱"
    if p > ma20: return "H1 多頭回歸"
    if p < lw * 0.95: return "L3 恐慌崩盤"
    if p < lw: return "L2 超賣區"
    return "L1 震盪整理"

def get_cfo_directive_v3(p_now, six_state, trend_status, range_high, range_low, mvrv_z, rsi, slope, vol_ratio):
    if "L" in six_state and "空頭" in trend_status: return "⬛ 趨勢損毀 (清倉)", 0.0
    if ("H3" in six_state) or (rsi is not None and rsi > 80): return "🟥 極限噴出 (賣1/2)", 0.0
    if range_high > 0 and p_now >= range_high: return "🟥 達預測高點 (賣1/2)", 0.0
    if "H2" in six_state: return "🟧 過熱減碼 (賣1/3)", 0.0
    
    buy_signals = []; build_pct = 0.0
    if (mvrv_z is not None and mvrv_z < -0.5) or (range_low > 0 and p_now < range_low): buy_signals.append("🔵 價值買點"); build_pct = max(build_pct, 0.5)
    if "L2" in six_state: buy_signals.append("💎 抄底機會 (30%)"); build_pct = max(build_pct, 0.3)
    if "多頭" in trend_status and ("H1" in six_state or "L1" in six_state):
        if slope is not None and slope > 0.01 and vol_ratio > 1.5: buy_signals.append("🔥 加速進攻 (80%)"); build_pct = max(build_pct, 0.8)
        elif slope is not None and slope > 0: buy_signals.append("🟢 多頭確立 (50%)"); build_pct = max(build_pct, 0.5)
        else: buy_signals.append("🟢 轉強試單 (20%)"); build_pct = max(build_pct, 0.2)
    
    msg = " | ".join(buy_signals) if buy_signals else "⬜ 觀望/持有"
    return msg, build_pct

def analyze_trend_multi(series):
    if series is None or len(series) < 126: return {}
    y = series.iloc[-126:].values.reshape(-1, 1); x = np.arange(len(y)).reshape(-1, 1); model = LinearRegression().fit(x, y)
    p_now = series.iloc[-1]; sma200 = series.rolling(200).mean().iloc[-1]; status = "🔥 多頭" if p_now > sma200 else "🛑 空頭"
    if p_now < sma200 and p_now > sma200 * 0.9: status = "📉 弱勢"
    return {"p_1m": model.predict([[len(y)+22]])[0].item(), "p_now": p_now, "status": status}

def calc_dynamic_kelly(series, lookback=63):
    try:
        if len(series) < lookback: return 0.0
        recent = series.iloc[-lookback:]; rets = recent.pct_change().dropna(); wins = rets[rets > 0]; losses = rets[rets < 0]
        if len(losses) == 0: return 1.0 
        if len(wins) == 0: return 0.0
        win_rate = len(wins) / len(rets); avg_win = wins.mean(); avg_loss = abs(losses.mean()); wl_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        kelly = win_rate - ((1 - win_rate) / wl_ratio); return max(0.0, min(1.0, kelly * 0.5))
    except: return 0.0

def calc_obv(close, volume):
    return (np.sign(close.diff()) * volume).fillna(0).cumsum() if volume is not None else None

def compare_with_leverage(ticker, df_close):
    benchs = ['QQQ', 'QLD', 'TQQQ']; valid_benchs = [b for b in benchs if b in df_close.columns]
    if not valid_benchs or ticker not in df_close.columns: return None
    lookback = min(len(df_close), 252); df_compare = df_close[[ticker] + valid_benchs].iloc[-lookback:].copy()
    df_norm = df_compare / df_compare.iloc[0] * 100; ret_ticker = df_norm[ticker].iloc[-1] - 100; ret_tqqq = df_norm['TQQQ'].iloc[-1] - 100 if 'TQQQ' in df_norm else 0
    return df_norm, "👑 贏 TQQQ" if ret_ticker > ret_tqqq else "💀 輸 TQQQ", ret_ticker, ret_tqqq

# [STRATEGY ENGINE]
def run_strategy_backtest_pro(df_in, vol_in, frequency_days=1):
    df = df_in.copy(); df['Volume'] = vol_in
    if len(df) > 300: df = df.iloc[-300:]
    elif len(df) < 250: return None, 0, 0, 0, 0, 0
    df['SMA20'] = df['Close'].rolling(20).mean(); df['SMA200'] = df['Close'].rolling(200).mean(); df['STD20'] = df['Close'].rolling(20).std(); df['Upper'] = df['SMA20'] + 2 * df['STD20']; df['Lower'] = df['SMA20'] - 2 * df['STD20']
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).ewm(13).mean() / df['Close'].diff().clip(upper=0).abs().ewm(13).mean())); df['Slope'] = (df['SMA20'] - df['SMA20'].shift(5)) / df['SMA20'].shift(5); df['Vol_MA'] = df['Volume'].rolling(20).mean(); df['Vol_Ratio'] = df['Volume'] / df['Vol_MA']; df['MVRV_Z'] = (df['Close'] - df['SMA200']) / df['Close'].rolling(200).std()
    cash_dca = 0; shares_dca = 0; invested = 0; cash_strat = 0; shares_strat = 0; history = []; last_month = -1
    for i in range(len(df)):
        p = df['Close'].iloc[i]; date = df.index[i]
        if date.month != last_month: cash_dca += 10000; cash_strat += 10000; invested += 10000; last_month = date.month; buy = cash_dca // p; shares_dca += buy; cash_dca -= buy * p
        if i % frequency_days == 0 and i > 20:
            ma20 = df['SMA20'].iloc[i]; up = df['Upper'].iloc[i]; lw = df['Lower'].iloc[i]; ma200 = df['SMA200'].iloc[i]; rsi = df['RSI'].iloc[i]; slope = df['Slope'].iloc[i]; vol_r = df['Vol_Ratio'].iloc[i]; mvrv_z = df['MVRV_Z'].iloc[i]
            sell_pct = 0
            if p < ma20 and ma200 > 0 and p < ma200: sell_pct = 1.0 
            elif p > up * 1.05 or rsi > 80: sell_pct = 0.5 
            elif p > up: sell_pct = 0.33 
            if sell_pct > 0 and shares_strat > 0: s_amt = int(shares_strat * sell_pct); shares_strat -= s_amt; cash_strat += s_amt * p
            buy_pct = 0
            if sell_pct == 0:
                if mvrv_z < -0.5: buy_pct = 0.5 
                elif p < lw: buy_pct = 0.3 
                elif p > ma20 and p > ma200 and p < up:
                    if slope > 0.01 and vol_r > 1.5: buy_pct = 0.8 
                    elif slope > 0: buy_pct = 0.5 
                    else: buy_pct = 0.2 
            if buy_pct > 0 and cash_strat > 1000: b_val = cash_strat * buy_pct; buy = b_val // p; shares_strat += buy; cash_strat -= buy * p
        history.append({"Date": date, "DCA": cash_dca + shares_dca * p, "Strat": cash_strat + shares_strat * p, "Inv": invested})
    res = pd.DataFrame(history).set_index("Date"); return res, (res['DCA'].iloc[-1]-invested)/invested, (res['Strat'].iloc[-1]-invested)/invested, invested, res['DCA'].iloc[-1], res['Strat'].iloc[-1]

# --- 3. 財務 ---
def run_traffic_light(series):
    sma200 = series.rolling(200).mean(); df = pd.DataFrame({'Close': series, 'SMA200': sma200}); df['Signal'] = np.where(df['Close'] > df['SMA200'], 1, 0); df['Strategy'] = (1 + df['Close'].pct_change() * df['Signal'].shift(1)).cumprod(); df['BuyHold'] = (1 + df['Close'].pct_change()).cumprod(); return df['Strategy'], df['BuyHold']

def parse_input(text):
    port = {}
    for line in text.strip().split('\n'):
        if ',' in line:
            parts = line.split(','); 
            try: 
                port[parts[0].strip().upper()] = float(parts[1].strip())
            except: 
                port[parts[0].strip().upper()] = 0.0
    return port

# --- MAIN APP ---
def main():
    with st.sidebar:
        st.header("⚙️ 指揮官錢包")
        fred_key = st.secrets.get("FRED_API_KEY", st.text_input("FRED API Key", type="password"))
        user_cash = st.number_input("💰 現金儲備 (USD)", value=10000.0, step=1000.0)
        default_input = "BTC-USD, 10000\nAMD, 10000\nNVDA, 10000\nTLT, 5000\nURA, 5000"
        user_input = st.text_area("持倉市值清單", default_input, height=150)
        portfolio_dict = parse_input(user_input); tickers_list = list(portfolio_dict.keys())
        stock_value = sum(portfolio_dict.values())
        total_assets = user_cash + stock_value
        st.metric("🏦 總資產", f"${total_assets:,.0f}", f"現金: ${user_cash:,.0f}")
        slot_limit = st.slider("單一持股預算上限 (%)", 5, 50, 20) / 100
        if st.button("🚀 啟動戰略指揮中心", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False): return

    with st.spinner("🦅 Alpha 12.6 正在擬定戰略指令..."):
        df_close, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_macro, df_fed = fetch_fred_macro(fred_key); adv_data = {t: get_advanced_info(t) for t in tickers_list}

    if df_close.empty: st.error("❌ 無數據"); st.stop()

    t1, t2, t3, t4, t5, t6, t7 = st.tabs(["🦅 戰略戰情", "🐋 深度籌碼", "🔍 個股體檢", "🚦 策略回測", "💰 CFO 財報", "🏠 房貸目標", "📊 策略實驗室"])

    with t1:
        st.subheader("1. 宏觀與總表")
        liq = df_macro['Net_Liquidity'].iloc[-1] if df_macro is not None else 0; vix = df_close['^VIX'].iloc[-1]; tnx = df_close['^TNX'].iloc[-1]
        try: cg = (df_close['HG=F'].iloc[-1]/df_close['GC=F'].iloc[-1])*1000
        except: cg = 0
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("💧 淨流動性", f"${liq:.2f}T"); c2.metric("🌪️ VIX", f"{vix:.2f}", delta_color="inverse"); c3.metric("⚖️ 10年債", f"{tnx:.2f}%"); c4.metric("🏭 銅金比", f"{cg:.2f}"); c5.metric("🏦 Fed利率", f"{df_fed['Fed_Rate'].iloc[-1]:.2f}%" if df_fed is not None else "N/A")
        
        st.markdown("#### 📊 CFO 戰略指令總表 (含建倉建議金額)")
        summary = []
        for t in tickers_list:
            if t not in df_close.columns: continue
            trend = analyze_trend_multi(df_close[t]); targets = calc_targets_composite(t, df_close, df_high, df_low, adv_data.get(t,{}), 22)
            six_s = calc_six_dim_state(df_close[t]); d_kelly = calc_dynamic_kelly(df_close[t], 63); mvrv_s = calc_mvrv_z(df_close[t]); mvrv_z = mvrv_s.iloc[-1] if mvrv_s is not None else 0
            rsi, slope, vol_r = calc_tech_indicators(df_close[t], df_vol[t])
            vol_d = df_close[t].pct_change().std(); price_s = df_close[t].iloc[-1] * vol_d * np.sqrt(22); tgt = targets['Avg'] if targets and targets['Avg'] else 0
            # 計算建倉建議
            cfo_act, build_pct = get_cfo_directive_v3(trend['p_now'], six_s, trend['status'], tgt+2*price_s, tgt-2*price_s, mvrv_z, rsi, slope, vol_r)
            
            # [錢包邏輯] 
            stock_budget = total_assets * slot_limit
            # 參考 Kelly: 如果 Kelly 建議更小，則縮小預算
            effective_budget = min(stock_budget, total_assets * d_kelly) if d_kelly > 0 else stock_budget
            suggested_pos = effective_budget * build_pct
            current_pos = portfolio_dict.get(t, 0)
            to_buy = max(0, suggested_pos - current_pos)
            
            summary.append({"代號": t, "現價": f"${trend['p_now']:.2f}", "CFO 指令": cfo_act, "建議建倉(USD)": f"${to_buy:,.0f}" if build_pct > 0 else "-", "預計總額": f"${suggested_pos:,.0f}" if build_pct > 0 else "-", "Kelly": f"{d_kelly*100:.1f}%", "預測值": f"${tgt:.2f}"})
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
        st.caption(f"💡 建議金額基於：單一持股預算上限 ${total_assets * slot_limit:,.0f} (總資產的 {slot_limit*100:.0f}%) 與動態凱利保守值。")

        st.markdown("---")
        st.subheader("2. 個股戰略雷達 (詳細分析)")
        for t in tickers_list:
            if t not in df_close.columns: continue
            targets = calc_targets_composite(t, df_close, df_high, df_low, adv_data.get(t,{}), 22)
            bt = run_backtest_lab(t, df_close, df_high, df_low, 22); obv = calc_obv(df_close[t], df_vol[t]); comp_res = compare_with_leverage(t, df_close); six_s = calc_six_dim_state(df_close[t])
            with st.expander(f"🦅 {t} | {six_s} | 目標: ${targets['Avg']:.2f}", expanded=False):
                k1, k2, k3 = st.columns([2, 1, 1])
                with k1: 
                    if comp_res: st.plotly_chart(px.line(comp_res[0], title=f"{t} vs TQQQ").update_layout(height=300), use_container_width=True)
                with k2:
                    st.markdown("#### 🎯 五角定位")
                    for k, v in targets.items(): st.write(f"**{k}:** ${v:.2f}" if v else f"**{k}:** N/A")
                    if bt: st.info(f"回測誤差: {bt['Error']:.1%}")
                with k3:
                    st.markdown("#### 📉 資金流 (OBV)")
                    fig = go.Figure(); fig.add_trace(go.Scatter(y=df_close[t].iloc[-126:], name='Price'))
                    if obv is not None: fig.add_trace(go.Scatter(y=obv.iloc[-126:], name='OBV', yaxis='y2'))
                    fig.update_layout(height=300, yaxis2=dict(overlaying='y', side='right')); st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.subheader("🐋 籌碼與內部人")
        chip_data = []
        for t in tickers_list:
            if t not in df_close.columns: continue
            info = adv_data.get(t, {})
            chip_data.append({
                "代號": t, 
                "機構持股": f"{(info.get('Inst_Held') or 0)*100:.1f}%", 
                "內部人": f"{(info.get('Insider_Held') or 0)*100:.1f}%", 
                "空單": f"{(info.get('Short_Ratio') or 0):.2f}"
            })
        st.dataframe(pd.DataFrame(chip_data), use_container_width=True)
    with t3:
        st.subheader("🔍 財務體質")
        h_data = []
        for t in tickers_list:
            info = adv_data.get(t, {})
            h_data.append({
                "代號": t, 
                "PEG": f"{(info.get('PEG') or 0):.2f}", 
                "ROE": f"{(info.get('ROE') or 0)*100:.1f}%", 
                "淨利率": f"{(info.get('Profit_Margin') or 0)*100:.1f}%", 
                "流動比": info.get('Current_Ratio'), 
                "負債/權益": info.get('Debt_Equity')})
        st.dataframe(pd.DataFrame(h_data), use_container_width=True)
    with t4:
        st.subheader("🚦 策略回測 (SMA200)")
        for t in tickers_list:
            if t in df_close.columns: s, b = run_traffic_light(df_close[t]); st.write(f"**{t}**"); st.line_chart(pd.concat([s, b], axis=1))
    with t5:
        st.subheader("💰 CFO 財報")
        inc = st.number_input("月收", 80000); exp = st.number_input("月支", 40000); st.metric("儲蓄率", f"{(inc-exp)/inc:.1%}")
    with t6:
        st.subheader("🏠 房貸目標")
def calc_mortgage(amt, yrs, rate):
    	# amt: 貸款總額, yrs: 貸款年限, rate: 年利率 (%)
    	r = rate / 100 / 12  # 月利率
    	m = yrs * 12         # 總期數
    	if r > 0:
        pmt = amt * (r * (1 + r)**m) / ((1 + r)**m - 1)
    	else:
        pmt = amt / m
return pmt, pmt * m - amt
        amt = st.number_input("貸", 10000000); rt = st.number_input("率", 2.2); pmt, _ = calc_mortgage(amt, 30, rt); st.metric("月付", f"${pmt:,.0f}")
    with t7:
        st.subheader("📊 策略實驗室")
        avail = [c for c in df_close.columns if not (c.startswith('^') or c.endswith('=F'))]
        lab_t = st.selectbox("選擇實驗標的", sorted(list(set(avail + ['TQQQ', 'QQQ', 'SPY']))))
        if lab_t in df_close.columns:
            res1, r1, sr1, inv, dca1, st1 = run_strategy_backtest_pro(df_close[lab_t].to_frame(name='Close'), df_vol[lab_t], 1)
            res3, r3, sr3, _, _, st3 = run_strategy_backtest_pro(df_close[lab_t].to_frame(name='Close'), df_vol[lab_t], 3)
            if res1 is not None:
                k1, k2, k3 = st.columns(3); k1.metric("投入本金", f"${inv:,.0f}"); k2.metric("DCA 淨值", f"${dca1:,.0f}", f"{r1:.1%}"); k3.metric("策略(1D) 淨值", f"${st1:,.0f}", f"{sr1:.1%}")
                fig = go.Figure(); fig.add_trace(go.Scatter(x=res1.index, y=res1['DCA'], name='DCA', line=dict(color='gray', dash='dash'))); fig.add_trace(go.Scatter(x=res1.index, y=res1['Strat'], name='CFO Strategy', line=dict(color='#00BFFF'))); st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()