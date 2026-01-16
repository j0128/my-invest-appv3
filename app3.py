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
st.set_page_config(page_title="Alpha 13.0: 混合指揮官", layout="wide", page_icon="🦅")

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

# --- 1. 數據引擎 ---

@st.cache_data(ttl=1800)
def fetch_market_data(tickers):
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
    except Exception: return None, None

@st.cache_data(ttl=3600*24)
def get_advanced_info(ticker):
    try:
        t = yf.Ticker(ticker); info = t.info
        peg = info.get('pegRatio'); fwd_pe = info.get('forwardPE'); earn_growth = info.get('earningsGrowth')
        if peg is None and fwd_pe is not None and earn_growth is not None and earn_growth > 0:
            peg = fwd_pe / (earn_growth * 100)
        return {
            'Type': 'ETF' if 'ETF' in info.get('quoteType', '').upper() else 'Stock',
            'Target_Mean': info.get('targetMeanPrice'), 'Forward_PE': fwd_pe, 'PEG': peg,
            'Inst_Held': info.get('heldPercentInstitutions'), 'Insider_Held': info.get('heldPercentInsiders'),
            'Short_Ratio': info.get('shortRatio'), 'Current_Ratio': info.get('currentRatio'),
            'Debt_Equity': info.get('debtToEquity'), 'ROE': info.get('returnOnEquity'),
            'Profit_Margin': info.get('profitMargins'),
            'trailingEps': info.get('trailingEps'), 'forwardEps': info.get('forwardEps'),
            'trailingPE': info.get('trailingPE')
        }
    except Exception: return {}

# --- 2. 戰略與估值模型 ---

def train_rf_model(df_close, ticker, days_forecast=22):
    try:
        if ticker not in df_close.columns: return None
        df = pd.DataFrame(index=df_close.index); df['Close'] = df_close[ticker]
        df['Ret'] = df['Close'].pct_change(); df['Vol'] = df['Ret'].rolling(20).std()
        df['SMA'] = df['Close'].rolling(20).mean(); df['Target'] = df['Close'].shift(-days_forecast)
        df = df.dropna()
        if len(df) < 60: return None
        X = df.drop(columns=['Target', 'Close']); y = df['Target']
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42); model.fit(X, y)
        return model.predict(X.iloc[[-1]])[0]
    except Exception: return None

def calc_targets_composite(ticker, df_close, df_high, df_low, f_data, days_forecast=22):
    if ticker not in df_close.columns: return None
    c = df_close[ticker]; h = df_high[ticker]; l = df_low[ticker]
    try:
        # 1. 技術面預測
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        t_atr = c.iloc[-1] + (tr.rolling(14).mean().iloc[-1] * np.sqrt(days_forecast))
        t_mc = c.iloc[-1] * ((1 + c.pct_change().mean())**days_forecast)
        t_fib = c.iloc[-60:].max() + (c.iloc[-60:].max() - c.iloc[-60:].min()) * 0.618 
        t_rf = train_rf_model(df_close, ticker, days_forecast)
        
        # 2. 財務面預測 (DCF & PE Target)
        eps = f_data.get('trailingEps') or 0
        fwd_eps = f_data.get('forwardEps') or 0
        t_pe = (fwd_eps * f_data.get('trailingPE')) if f_data.get('trailingPE') else None
        
        # 簡易 DCF: EPS * (1+G)^5 * PE / (1.1^5)
        growth = (f_data.get('PEG') or 1.5) / 10 # 假設
        t_dcf = (eps * (1 + growth)**5 * (f_data.get('trailingPE') or 20)) / (1.1**5) if eps > 0 else None

        targets = [t for t in [t_atr, t_mc, t_fib, f_data.get('Target_Mean'), t_rf, t_pe, t_dcf] if t is not None and not pd.isna(t)]
        return {"Avg": sum(targets) / len(targets) if targets else None, "ATR": t_atr, "MC": t_mc, "Fib": t_fib, "RF": t_rf, "PE_Target": t_pe, "DCF": t_dcf}
    except Exception: return None

# [預測值回測引擎]
def run_backtest_lab(ticker, df_close, df_high, df_low, f_data, days_ago=22):
    if ticker not in df_close.columns or len(df_close) < 250: return None
    idx_past = len(df_close) - days_ago - 1
    p_now = df_close[ticker].iloc[-1]
    df_past = df_close.iloc[:idx_past+1]
    # 使用過去數據重新計算
    targets_past = calc_targets_composite(ticker, df_past, df_high.iloc[:idx_past+1], df_low.iloc[:idx_past+1], f_data, days_ago)
    past_pred = targets_past['Avg'] if targets_past else None
    err = (past_pred - p_now) / p_now if past_pred else 0
    return {"Past_Pred": past_pred, "Error": err}

def analyze_trend_multi(series):
    if len(series) < 200: return {"status": "資料不足", "p_now": series.iloc[-1], "is_bull": False}
    p_now = series.iloc[-1]; sma200 = series.rolling(200).mean().iloc[-1]
    sma200_prev = series.rolling(200).mean().iloc[-10]
    is_bull = (p_now > sma200) and (sma200 > sma200_prev)
    return {"status": "🔥 多頭" if p_now > sma200 else "🛑 空頭", "p_now": p_now, "sma200": sma200, "is_bull": is_bull}

def get_cfo_directive_v4(p_now, six_state, trend_status, bull_mode, rsi, slope, vol_ratio, mvrv_z, range_high, range_low):
    if "L" in six_state and "空頭" in trend_status: return "⬛ 趨勢損毀 (清倉)", 0.0
    rsi_limit = 85 if bull_mode else 80
    if ("H3" in six_state) or (rsi is not None and rsi > rsi_limit): return "🟥 極限噴出 (賣1/2)", 0.5
    if not bull_mode:
        if range_high > 0 and p_now >= range_high: return "🟥 達預測高點 (賣1/2)", 0.5
        if "H2" in six_state: return "🟧 過熱減碼 (賣1/3)", 0.66
    buy_signals = []; build_pct = 0.5 if bull_mode else 0.0
    if (mvrv_z is not None and mvrv_z < -0.5) or (range_low > 0 and p_now < range_low): 
        buy_signals.append("🔵 價值買點"); build_pct = max(build_pct, 0.5)
    if "L2" in six_state: buy_signals.append("💎 抄底機會"); build_pct = max(build_pct, 0.3)
    if "多頭" in trend_status:
        if slope is not None and slope > 0.01 and vol_ratio > 1.5: buy_signals.append("🔥 加速進攻"); build_pct = max(build_pct, 0.8)
        elif slope is not None and slope > 0: buy_signals.append("🟢 多頭確立"); build_pct = max(build_pct, 0.5)
        else: buy_signals.append("🟢 轉強試單"); build_pct = max(build_pct, 0.2)
    return (" | ".join(buy_signals) if buy_signals else ("🦁 牛市持倉" if bull_mode else "⬜ 觀望/持有")), build_pct

# --- 3. 財務計算與對稱功能 (完全還原) ---

def run_strategy_backtest_salary_flow_v2(df_in, vol_in):
    df = df_in.copy(); df['Volume'] = vol_in
    if len(df) > 300: df = df.iloc[-300:]
    df['SMA20'] = df['Close'].rolling(20).mean(); df['SMA200'] = df['Close'].rolling(200).mean()
    df['Upper'] = df['SMA20'] + 2 * df['Close'].rolling(20).std(); df['Lower'] = df['SMA20'] - 2 * df['Close'].rolling(20).std()
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).ewm(13).mean() / df['Close'].diff().clip(upper=0).abs().ewm(13).mean()))
    cash_dca = 0; shares_dca = 0; cash_strat = 0; shares_strat = 0; invested = 0; history = []; last_month = -1
    for i in range(len(df)):
        p = df['Close'].iloc[i]; date = df.index[i]
        if date.month != last_month:
            cash_dca += 10000; cash_strat += 10000; invested += 10000; last_month = date.month
            buy_dca = cash_dca // p; shares_dca += buy_dca; cash_dca -= buy_dca * p
        if i > 20:
            ma20 = df['SMA20'].iloc[i]; ma200 = df['SMA200'].iloc[i]; bull = (p > ma200) and (ma200 > df['SMA200'].iloc[i-5]) and (p > ma20)
            rsi = df['RSI'].iloc[i]; up = df['Upper'].iloc[i]; lw = df['Lower'].iloc[i]
            sell_pct = 0
            if p < ma20 and p < ma200: sell_pct = 1.0
            elif p > up * 1.05 or rsi > (85 if bull else 80): sell_pct = 0.5
            if sell_pct > 0 and shares_strat > 0:
                s_amt = int(shares_strat * sell_pct); shares_strat -= s_amt; cash_strat += s_amt * p
            if sell_pct == 0:
                buy_pct = 0.8 if bull else (0.3 if p < lw else 0)
                if buy_pct > 0 and cash_strat > 100:
                    b_val = cash_strat * buy_pct; buy = b_val // p; shares_strat += buy; cash_strat -= buy * p
        history.append({"Date": date, "DCA": cash_dca + shares_dca * p, "Strat": cash_strat + shares_strat * p, "Inv": invested})
    res = pd.DataFrame(history).set_index("Date"); return res, (res['DCA'].iloc[-1]-invested)/invested, (res['Strat'].iloc[-1]-invested)/invested, invested, res['DCA'].iloc[-1], res['Strat'].iloc[-1]

def calc_mortgage(amt, yrs, rate):
    r = rate / 100 / 12; m = yrs * 12
    pmt = amt * (r * (1 + r)**m) / ((1 + r)**m - 1) if r > 0 else amt / m
    return pmt, pmt * m - amt

# --- 4. MAIN APP ---

def main():
    with st.sidebar:
        st.header("⚙️ 指揮系統設定")
        fred_key = st.secrets.get("FRED_API_KEY", st.text_input("FRED API Key", type="password"))
        user_cash = st.number_input("💰 現金儲備 (USD)", value=10000.0, step=1000.0)
        user_input = st.text_area("持倉市值清單", "BTC-USD, 10000\nAMD, 10000\nNVDA, 10000", height=150)
        p_dict = {line.split(',')[0].strip().upper(): float(line.split(',')[1].strip()) for line in user_input.strip().split('\n') if ',' in line}
        tickers_list = list(p_dict.keys())
        total_assets = user_cash + sum(p_dict.values())
        st.metric("🏦 總資產", f"${total_assets:,.0f}", f"現金: ${user_cash:,.0f}")
        slot_limit = st.slider("預算上限 (%)", 5, 50, 20) / 100
        if st.button("🚀 啟動 Alpha 指揮中心", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False): return

    with st.spinner("🦅 Alpha 13.0 正在掃描全域戰場..."):
        df_close, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_macro, df_fed = fetch_fred_macro(fred_key); adv_data = {t: get_advanced_info(t) for t in tickers_list}

    t1, t2, t3, t4, t5, t6, t7 = st.tabs(["🦅 戰略戰情", "🐋 深度籌碼", "🔍 個股體檢", "🚦 策略回測", "💰 CFO 財報", "🏠 房貸目標", "📊 策略實驗室"])

    with t1:
        st.title("🦅 Alpha 13.0: 混合戰略指揮中心")
        st.subheader("1. 宏觀趨勢")
        liq = df_macro['Net_Liquidity'].iloc[-1] if df_macro is not None else 0
        vix = df_close['^VIX'].iloc[-1]; tnx = df_close['^TNX'].iloc[-1]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("💧 淨流動性", f"${liq:.2f}T"); c2.metric("🌪️ VIX", f"{vix:.2f}", delta_color="inverse")
        c3.metric("⚖️ 10年債", f"{tnx:.2f}%"); c4.metric("🏭 銅金比", "N/A"); c5.metric("🏦 Fed利率", f"{df_fed['Fed_Rate'].iloc[-1]:.2f}%" if df_fed is not None else "N/A")
        
        st.markdown("#### 📊 CFO 混合戰略總表 (含 $±2\sigma$ 預測區間)")
        summary = []
        for t in tickers_list:
            if t not in df_close.columns: continue
            tr = analyze_trend_multi(df_close[t]); targets = calc_targets_composite(t, df_close, df_high, df_low, adv_data.get(t,{}), 22)
            tgt = targets['Avg'] if targets else 0
            vol_22 = df_close[t].pct_change().std() * np.sqrt(22)
            pred_range = f"${tr['p_now']*(1-2*vol_22):.2f} - ${tr['p_now']*(1+2*vol_22):.2f}"
            
            # 指標計算
            rsi = 100 - (100 / (1 + df_close[t].diff().clip(lower=0).ewm(13).mean() / df_close[t].diff().clip(upper=0).abs().ewm(13).mean())).iloc[-1]
            ma20 = df_close[t].rolling(20).mean(); slope = (ma20.iloc[-1] - ma20.iloc[-5]) / ma20.iloc[-5]
            vol_r = df_vol[t].iloc[-1] / df_vol[t].rolling(20).mean().iloc[-1]
            mvrv_z = (tr['p_now'] - tr['sma200']) / df_close[t].rolling(200).std().iloc[-1]
            cfo_act, b_pct = get_cfo_directive_v4(tr['p_now'], "H1", tr['status'], tr['is_bull'], rsi, slope, vol_r, mvrv_z, tgt*1.05, tgt*0.95)
            
            summary.append({"代號": t, "模式": "BULL" if tr['is_bull'] else "BEAR", "現價": f"${tr['p_now']:.2f}", "CFO 指令": cfo_act, "預期範圍(±2σ)": pred_range, "目標價(Avg)": f"${tgt:.2f}"})
        st.write(pd.DataFrame(summary).to_html(escape=False), unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("2. 個股雷達 (估值與回測)")
        for t in tickers_list:
            if t not in df_close.columns: continue
            targets = calc_targets_composite(t, df_close, df_high, df_low, adv_data.get(t,{}), 22)
            bt_err = run_backtest_lab(t, df_close, df_high, df_low, adv_data.get(t,{}), 22)
            with st.expander(f"🦅 {t} 戰略深度分析", expanded=False):
                k1, k2, k3 = st.columns([2, 1, 1])
                with k1: 
                    st.plotly_chart(px.line(df_close[t].iloc[-126:], title=f"{t} 趨勢圖"), use_container_width=True)
                with k2:
                    st.markdown("#### 🎯 估值體系 (1M)")
                    if targets:
                        for key, val in targets.items(): st.write(f"**{key}:** ${val:.2f}" if val else f"**{key}:** N/A")
                    if bt_err: st.info(f"預測值回測誤差: {bt_err['Error']:.1%}")
                with k3:
                    st.markdown("#### 🐋 籌碼與 Kelly")
                    st.write(f"機構持股: {(adv_data.get(t,{}).get('Inst_Held') or 0)*100:.1f}%")
                    st.write(f"動態 Kelly: {tr['p_now']/tr['sma200']-1:.1%}")

    with t2:
        st.subheader("🐋 深度籌碼")
        chip_data = [{"代號": t, "機構持股": f"{(adv_data.get(t,{}).get('Inst_Held') or 0)*100:.1f}%", "內部人": f"{(adv_data.get(t,{}).get('Insider_Held') or 0)*100:.1f}%"} for t in tickers_list]
        st.dataframe(pd.DataFrame(chip_data), use_container_width=True)

    with t3:
        st.subheader("🔍 財務體質")
        h_data = [{"代號": t, "PEG": f"{(adv_data.get(t,{}).get('PEG') or 0):.2f}", "ROE": f"{(adv_data.get(t,{}).get('ROE') or 0)*100:.1f}%", "淨利率": f"{(adv_data.get(t,{}).get('Profit_Margin') or 0)*100:.1f}%"} for t in tickers_list]
        st.dataframe(pd.DataFrame(h_data), use_container_width=True)

    with t6:
        st.subheader("🏠 房貸目標")
        amt = st.number_input("金額", 10000000); rt = st.number_input("利率", 2.2); pmt, _ = calc_mortgage(amt, 30, rt); st.metric("月付", f"${pmt:,.0f}")

    with t7:
        st.subheader("📊 混合策略實驗室")
        lab_ticker = st.selectbox("標的", sorted(list(set(tickers_list + ['TQQQ', 'SPY']))))
        if lab_ticker in df_close.columns:
            res, r_dca, r_strat, inv, dca_f, strat_f = run_strategy_backtest_salary_flow_v2(df_close[lab_ticker].to_frame(name='Close'), df_vol[lab_ticker])
            c1, c2, c3 = st.columns(3); c1.metric("本金", f"${inv:,.0f}"); c2.metric("DCA", f"{r_dca:.1%}"); c3.metric("混合策略", f"{r_strat:.1%}")
            st.plotly_chart(px.line(res[['DCA', 'Strat']], title="淨值對比"), use_container_width=True)

if __name__ == "__main__":
    main()