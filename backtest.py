"""台股強勢股篩選 回測 (Python 版)

規則 (預設值, 都可以用環境變數調整):
  1. 每天收盤後, 用和 scan.py 一樣的條件篩選 (52週新高 / 多頭排列 / 成交量 / 漲幅比)
  2. 隔天 (T+1): 前一天篩選出來的股票, 如果當天收盤上漲, 且收盤價在 5MA(週線)之上 -> 以當天收盤價買進
  3. 買進同時掛 10% 移動停損 (從買進後的最高價回落 10% 出場), 從買進的隔天開始生效
  4. 每天重複 2~3; 本金 100 萬, 最多同時持有 MAX_POSITIONS 檔, 每檔資金 = 當時總資產 / MAX_POSITIONS

輸出: backtest_summary.md / backtest_trades.csv / backtest_equity.csv / backtest_equity.png
"""
import os
import time
from io import StringIO
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

TZ = timezone(timedelta(hours=8))
TODAY = datetime.now(TZ).date()


def _env(name, default, cast=float):
    v = os.getenv(name)
    return cast(v) if v not in (None, "") else default


START_DATE = os.getenv("START_DATE") or (TODAY - timedelta(days=365)).isoformat()   # 回測開始日
END_DATE = os.getenv("END_DATE") or TODAY.isoformat()                                # 回測結束日
INIT_CAPITAL = _env("INIT_CAPITAL", 1_000_000)      # 本金
MAX_POSITIONS = _env("MAX_POSITIONS", 10, int)      # 最多同時持有幾檔
STOP_PCT = _env("STOP_PCT", 10)                     # 移動停損 %
NEW_HIGH_DAYS = _env("NEW_HIGH_DAYS", 20, int)      # 幾天內創52週新高
BULL_DAYS = _env("BULL_DAYS", 3, int)               # 多頭排列至少天數
MIN_LOTS = _env("MIN_LOTS", 1000, int)              # 成交量(張)大於
GAIN_RATIO = _env("GAIN_RATIO", 120)                # 收盤 >= 多頭排列第一天收盤的 N%, 0=不限
FEE_DISCOUNT = _env("FEE_DISCOUNT", 1.0)            # 手續費折扣 (1 = 無折扣 0.1425%, 0.6 = 6折)
FEE_RATE = 0.001425 * FEE_DISCOUNT                  # 買、賣各收一次
TAX_RATE = _env("TAX_RATE", 0.003)                  # 證交稅, 只有賣出收
LOT_SIZE = _env("LOT_SIZE", 1, int)                 # 1 = 可買零股; 1000 = 只買整張
BENCHMARK = os.getenv("BENCHMARK") or "^TWII"       # 比較用大盤
BATCH = 100
HEADERS = {"User-Agent": "Mozilla/5.0"}


# ===================== 資料 =====================
def load_universe():
    """從證交所 ISIN 公告取得上市(.TW)、上櫃(.TWO)普通股清單"""
    rows = []
    for mode, suffix in [(2, ".TW"), (4, ".TWO")]:
        url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
        html = requests.get(url, headers=HEADERS, timeout=30).content.decode("ms950", errors="ignore")
        df = pd.read_html(StringIO(html))[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        first = df.columns[0]
        df = df[df["CFICode"].astype(str).str.startswith("ES")]
        for s in df[first].astype(str):
            if "\u3000" not in s:
                continue
            code, name = s.split("\u3000", 1)
            code = code.strip()
            if code.isdigit() and len(code) == 4:
                rows.append((code + suffix, code, name.strip()))
    return pd.DataFrame(rows, columns=["ticker", "代號", "名稱"]).drop_duplicates("ticker")


def _norm_index(df):
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    return df


def download_all(tickers, start, end):
    """下載全部股票日K (還原權息價, 往前多抓一年做暖機: 52週高點、60MA 需要)"""
    import yfinance as yf
    warm = (pd.Timestamp(start) - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
    end_x = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")   # end 不含當天
    out = {}
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        print(f"⏳ 下載第 {i + 1}~{min(i + BATCH, len(tickers))} 檔 / {len(tickers)}", flush=True)
        data = None
        for _ in range(2):
            try:
                data = yf.download(batch, start=warm, end=end_x, interval="1d", group_by="ticker",
                                   auto_adjust=True, threads=True, progress=False)
                break
            except Exception as e:
                print(f"  ❌ 批次失敗: {e}", flush=True)
                time.sleep(3)
        if data is None:
            continue
        for tk in batch:
            try:
                d = data[tk] if isinstance(data.columns, pd.MultiIndex) else data
                d = d[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
            except Exception:
                continue
            if len(d) >= 130:
                out[tk] = _norm_index(d)
    return out


def load_benchmark(start, end):
    try:
        import yfinance as yf
        end_x = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        b = yf.download(BENCHMARK, start=start, end=end_x, interval="1d", auto_adjust=True, progress=False)
        c = b["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        c = _norm_index(c.dropna().to_frame("Close"))["Close"]
        return c if len(c) > 1 else None
    except Exception as e:
        print(f"  ⚠ 大盤資料下載失敗: {e}", flush=True)
        return None


# ===================== 篩選條件 (逐日, 向量化) =====================
def prepare(df):
    """對單一股票算出每一天的: 是否通過篩選(sig)、多頭排列天數(streak)、當天是否收漲(up)、收盤是否在5MA之上(above)"""
    c, h, v = df["Close"], df["High"], df["Volume"]
    n = len(df)
    cv = c.values

    roll = h.rolling(252, min_periods=200).max()
    flag = (h >= roll).astype(float)                                   # 當天創52週新高(盤中最高價)
    newhigh = flag.rolling(NEW_HIGH_DAYS, min_periods=1).max().values > 0

    ma5, ma10, ma20, ma60 = (c.rolling(k).mean() for k in (5, 10, 20, 60))
    bull = ((ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60)).values
    streak = np.zeros(n, dtype=int)
    run = 0
    for i, b in enumerate(bull):
        run = run + 1 if b else 0
        streak[i] = run

    idx = np.arange(n)
    start_price = np.where(streak > 0, cv[np.clip(idx - streak + 1, 0, n - 1)], np.nan)  # 多頭排列第一天的收盤

    ok = (v.values / 1000 > MIN_LOTS) & newhigh & (streak >= BULL_DAYS) & (idx >= 129)
    if GAIN_RATIO > 0:
        with np.errstate(invalid="ignore"):
            ok &= cv >= start_price * GAIN_RATIO / 100
    up = np.r_[False, cv[1:] > cv[:-1]]
    above = (c >= ma5).values
    return {"sig": ok, "streak": streak, "up": up, "above": above}


def build_arrays(data, start, end):
    tickers = list(data)
    prep = {tk: prepare(df) for tk, df in data.items()}
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(df.index) for df in data.values()])))
    master = all_dates[(all_dates >= pd.Timestamp(start) - pd.Timedelta(days=10)) & (all_dates <= pd.Timestamp(end))]
    start_idx = int(np.searchsorted(master.values, np.datetime64(pd.Timestamp(start))))
    T, N = len(master), len(tickers)
    O, H, L, C = (np.full((T, N), np.nan) for _ in range(4))
    SIG, UP, ABOVE = (np.zeros((T, N), dtype=bool) for _ in range(3))
    STREAK = np.zeros((T, N), dtype=int)
    for j, tk in enumerate(tickers):
        df, p = data[tk], prep[tk]
        pos = df.index.get_indexer(master)
        ok = pos >= 0
        pp = pos[ok]
        O[ok, j] = df["Open"].values[pp]
        H[ok, j] = df["High"].values[pp]
        L[ok, j] = df["Low"].values[pp]
        C[ok, j] = df["Close"].values[pp]
        SIG[ok, j] = p["sig"][pp]
        UP[ok, j] = p["up"][pp]
        ABOVE[ok, j] = p["above"][pp]
        STREAK[ok, j] = p["streak"][pp]
    return dict(master=master, tickers=tickers, start_idx=start_idx,
                O=O, H=H, L=L, C=C, SIG=SIG, UP=UP, ABOVE=ABOVE, STREAK=STREAK)


# ===================== 回測引擎 =====================
def simulate(A, names=None):
    names = names or {}
    master, tickers = A["master"], A["tickers"]
    O, H, L, C = A["O"], A["H"], A["L"], A["C"]
    SIG, UP, ABOVE, STREAK = A["SIG"], A["UP"], A["ABOVE"], A["STREAK"]
    T, N = C.shape
    cash = float(INIT_CAPITAL)
    pos = {}                     # j -> {shares, entry, peak, t0, cost}
    lastc = np.full(N, np.nan)   # 每檔最近一次收盤 (停牌時沿用)
    trades, eq_dates, eq_vals, invested = [], [], [], []

    def close_trade(j, t, price, reason, final=False):
        nonlocal cash
        p = pos.pop(j)
        gross = p["shares"] * price
        proceeds = gross if final else gross * (1 - FEE_RATE - TAX_RATE)
        if not final:
            cash += proceeds
        pnl = proceeds - p["cost"]
        trades.append({
            "代號": tickers[j].split(".")[0], "名稱": names.get(tickers[j], ""),
            "買進日": master[p["t0"]].strftime("%Y-%m-%d"), "買進價": round(p["entry"], 2), "股數": p["shares"],
            "賣出日": master[t].strftime("%Y-%m-%d"), "賣出價": round(price, 2), "出場原因": reason,
            "持有天數": int((master[t] - master[p["t0"]]).days),
            "損益(元)": round(pnl, 0), "報酬率(%)": round(pnl / p["cost"] * 100, 2),
        })

    for t in range(A["start_idx"], T):
        valid = ~np.isnan(C[t])
        lastc[valid] = C[t][valid]
        sold_today = set()

        # 1) 移動停損: 用「買進後到昨天為止的最高價」算停損價, 今天跳空或盤中跌破就出場
        for j in list(pos):
            if not valid[j]:
                continue
            p = pos[j]
            stop = p["peak"] * (1 - STOP_PCT / 100)
            if O[t, j] <= stop:
                close_trade(j, t, O[t, j], "跳空跌破停損")
                sold_today.add(j)
            elif L[t, j] <= stop:
                close_trade(j, t, stop, "移動停損")
                sold_today.add(j)
            else:
                p["peak"] = max(p["peak"], H[t, j])

        # 2) 買進: 前一天篩選出來、今天收漲、收盤在週線上 -> 以今天收盤買進
        if t >= 1 and len(pos) < MAX_POSITIONS:
            prev = t - 1
            cand = np.where(SIG[prev] & UP[t] & ABOVE[t] & valid)[0]
            cand = [j for j in cand if j not in pos and j not in sold_today]
            cand.sort(key=lambda j: (-STREAK[prev, j], tickers[j]))     # 多頭排列越久越優先
            cand = cand[:MAX_POSITIONS - len(pos)]
            if cand:
                equity_now = cash + sum(p["shares"] * lastc[j] for j, p in pos.items())
                alloc = equity_now / MAX_POSITIONS
                for j in cand:
                    price = C[t, j]
                    budget = min(alloc, cash)
                    shares = int(budget / (price * (1 + FEE_RATE)) // LOT_SIZE * LOT_SIZE)
                    if shares <= 0:
                        continue
                    cost = shares * price * (1 + FEE_RATE)
                    cash -= cost
                    pos[j] = {"shares": shares, "entry": price, "peak": price, "t0": t, "cost": cost}

        # 3) 收盤資產
        mv = sum(p["shares"] * lastc[j] for j, p in pos.items())
        eq_dates.append(master[t])
        eq_vals.append(cash + mv)
        invested.append(mv / (cash + mv) if cash + mv > 0 else 0)

    # 期末還沒出場的部位: 以最後收盤價估值 (不計賣出成本), 標示為「持有中」
    for j in list(pos):
        close_trade(j, T - 1, lastc[j], "持有中(未平倉)", final=True)

    eq = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_dates), name="equity")
    return pd.DataFrame(trades), eq, float(np.mean(invested)) if invested else 0.0


# ===================== 績效 =====================
def compute_metrics(eq, trades, avg_invested):
    m = {}
    m["期初資金"] = INIT_CAPITAL
    m["期末資產"] = float(eq.iloc[-1])
    m["總報酬率(%)"] = (eq.iloc[-1] / INIT_CAPITAL - 1) * 100
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    m["年化報酬率(%)"] = ((eq.iloc[-1] / INIT_CAPITAL) ** (365 / days) - 1) * 100
    dd = eq / eq.cummax() - 1
    m["最大回撤(%)"] = dd.min() * 100
    r = eq.pct_change().dropna()
    m["夏普值(年化, 無風險利率0)"] = float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 2 and r.std() > 0 else float("nan")
    m["平均持股水位(%)"] = avg_invested * 100
    closed = trades[trades["出場原因"] != "持有中(未平倉)"] if len(trades) else trades
    m["交易筆數(已出場)"] = len(closed)
    m["持有中筆數"] = int(len(trades) - len(closed))
    if len(closed):
        win = closed[closed["損益(元)"] > 0]
        loss = closed[closed["損益(元)"] <= 0]
        m["勝率(%)"] = len(win) / len(closed) * 100
        m["平均獲利(%)"] = win["報酬率(%)"].mean() if len(win) else 0.0
        m["平均虧損(%)"] = loss["報酬率(%)"].mean() if len(loss) else 0.0
        m["獲利因子"] = win["損益(元)"].sum() / abs(loss["損益(元)"].sum()) if len(loss) and loss["損益(元)"].sum() != 0 else float("inf")
        m["平均持有天數"] = closed["持有天數"].mean()
    return m


def monthly_table(eq):
    last = eq.groupby(eq.index.to_period("M")).last()
    prev = last.shift(1)
    prev.iloc[0] = INIT_CAPITAL
    return ((last / prev - 1) * 100).round(2)


def fmt(k, v):
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, float):
        if np.isinf(v):
            return "∞"
        if np.isnan(v):
            return "-"
        return f"{v:,.2f}" if ("元" not in k and k not in ("期初資金", "期末資產")) else f"{v:,.0f}"
    return str(v)


# ===================== 輸出 =====================
def save_outputs(trades, eq, metrics, bench):
    trades.sort_values(["買進日", "代號"]).to_csv("backtest_trades.csv", index=False, encoding="utf-8-sig")
    eq.rename("總資產").to_csv("backtest_equity.csv", encoding="utf-8-sig", index_label="日期")

    bench_ret = None
    bench_eq = None
    if bench is not None:
        b = bench[bench.index >= eq.index[0]]
        if len(b) > 1:
            bench_eq = b / b.iloc[0] * INIT_CAPITAL
            bench_ret = (b.iloc[-1] / b.iloc[0] - 1) * 100

    lines = ["# 台股強勢股回測結果", "",
             f"回測期間: {eq.index[0]:%Y-%m-%d} ~ {eq.index[-1]:%Y-%m-%d}　|　產生時間 (台灣): {datetime.now(TZ):%Y-%m-%d %H:%M}", "",
             "## 規則", "",
             f"- 篩選: {NEW_HIGH_DAYS} 天內創52週新高、5/10/20/60MA 多頭排列 ≥ {BULL_DAYS} 天、成交量 > {MIN_LOTS} 張、"
             f"收盤 ≥ 多頭排列第一天收盤的 {GAIN_RATIO:g}% (0=不限)",
             "- 隔天: 前一天篩選出來的股票, 當天收盤上漲且收盤價在 5MA 之上 → 以當天收盤價買進",
             f"- 買進後設 {STOP_PCT:g}% 移動停損 (從買進後最高價回落), 隔天開始生效; 每天重複",
             f"- 本金 {INIT_CAPITAL:,.0f} 元, 最多 {MAX_POSITIONS} 檔, 每檔約 1/{MAX_POSITIONS} 總資產; "
             f"手續費 {FEE_RATE * 100:.4f}% (買賣各一次) + 證交稅 {TAX_RATE * 100:.2f}% (賣出); 買進單位 {LOT_SIZE} 股", "",
             "## 績效", "", "| 指標 | 數值 |", "|---|---|"]
    for k, v in metrics.items():
        lines.append(f"| {k} | {fmt(k, v)} |")
    if bench_ret is not None:
        lines.append(f"| 同期大盤 ({BENCHMARK}) 報酬率(%) | {bench_ret:,.2f} |")

    mt = monthly_table(eq)
    lines += ["", "## 每月報酬率 (%)", "", "| 月份 | 報酬率 |", "|---|---|"]
    for k, v in mt.items():
        lines.append(f"| {k} | {v:+.2f}% |")

    if len(trades):
        closed = trades[trades["出場原因"] != "持有中(未平倉)"]
        cols = ["代號", "名稱", "買進日", "買進價", "賣出日", "賣出價", "出場原因", "報酬率(%)"]
        if len(closed):
            lines += ["", "## 最賺的 10 筆", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            for _, r in closed.sort_values("報酬率(%)", ascending=False).head(10).iterrows():
                lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
            lines += ["", "## 最虧的 10 筆", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            for _, r in closed.sort_values("報酬率(%)").head(10).iterrows():
                lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        opened = trades[trades["出場原因"] == "持有中(未平倉)"]
        if len(opened):
            lines += ["", "## 期末仍持有 (以最後收盤估值)", "", "| 代號 | 名稱 | 買進日 | 買進價 | 最後收盤 | 未實現報酬率 |", "|---|---|---|---|---|---|"]
            for _, r in opened.iterrows():
                lines.append(f"| {r['代號']} | {r['名稱']} | {r['買進日']} | {r['買進價']} | {r['賣出價']} | {r['報酬率(%)']:+.2f}% |")

    lines += ["", "## 注意事項", "",
              "- 股票清單用「現在」的上市櫃公司, 已下市的股票不在裡面 (存活者偏誤, 報酬會偏樂觀)。",
              "- 使用還原權息價; 沒有計入處置股的撮合限制、漲跌停買不到/賣不掉、滑價。",
              "- 處置股豁免成交量條件、市值篩選, 在回測中沒有套用 (沒有歷史處置資料)。",
              "- 買進以「當天收盤價」成交; 停損價在盤中觸及就以停損價成交, 跳空則以開盤價成交。",
              "- 期末未出場的部位以最後收盤估值, 不含賣出成本。"]
    open("backtest_summary.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eq.index, eq.values, label="Strategy")
        if bench_eq is not None:
            ax.plot(bench_eq.index, bench_eq.values, label=BENCHMARK, alpha=0.7)
        ax.axhline(INIT_CAPITAL, color="gray", lw=0.8, ls="--")
        ax.set_title("Equity curve")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("backtest_equity.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  ⚠ 畫圖失敗 (不影響結果): {e}", flush=True)
    return bench_ret


def run_backtest(data, names, start=None, end=None):
    start = start or START_DATE
    end = end or END_DATE
    A = build_arrays(data, start, end)
    trades, eq, avg_inv = simulate(A, names)
    return trades, eq, compute_metrics(eq, trades, avg_inv)


def main():
    t0 = time.time()
    print(f"回測期間 {START_DATE} ~ {END_DATE}，本金 {INIT_CAPITAL:,.0f}，最多 {MAX_POSITIONS} 檔，移動停損 {STOP_PCT:g}%", flush=True)
    uni = load_universe()
    names = dict(zip(uni["ticker"], uni["名稱"]))
    print(f"股票清單 {len(uni)} 檔，下載資料中...", flush=True)
    data = download_all(uni["ticker"].tolist(), START_DATE, END_DATE)
    print(f"取得 {len(data)} 檔資料 (耗時 {int(time.time() - t0)} 秒)，開始回測...", flush=True)
    trades, eq, metrics = run_backtest(data, names)
    bench = load_benchmark(START_DATE, END_DATE)
    bench_ret = save_outputs(trades, eq, metrics, bench)
    print("\n===== 回測結果 =====", flush=True)
    for k, v in metrics.items():
        print(f"{k}: {fmt(k, v)}", flush=True)
    if bench_ret is not None:
        print(f"同期大盤 ({BENCHMARK}) 報酬率(%): {bench_ret:,.2f}", flush=True)
    print(f"\n完成，總耗時 {int(time.time() - t0)} 秒。輸出: backtest_summary.md / backtest_trades.csv / backtest_equity.csv", flush=True)


if __name__ == "__main__":
    main()
