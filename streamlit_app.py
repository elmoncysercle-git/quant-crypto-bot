import json
import time
import pathlib
import yaml
import pandas as pd
import streamlit as st
import ccxt

st.set_page_config(page_title="Quant Crypto Bot", layout="wide")
st.title("🔁 Quant Crypto Rotation Bot — Dashboard")

# --- Load config/state paths ---
CFG_PATH = pathlib.Path("config.yml")
if not CFG_PATH.exists():
    st.error("config.yml not found. Add it to the repo root.")
    st.stop()

cfg = yaml.safe_load(CFG_PATH.read_text())
state_path = pathlib.Path(cfg.get("state_file", "state/state.json"))
symbols = cfg.get("trading", {}).get("symbols", [])
base_ccy = cfg.get("trading", {}).get("base_ccy", "USDT")
mode = cfg.get("mode", "paper")
rebalance_days = cfg.get("trading", {}).get("rebalance_days", 7)
cash_buf = cfg.get("trading", {}).get("cash_buffer_pct", 0.15)

st.subheader("Config")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mode", mode.upper())
c2.metric("Base CCY", base_ccy)
c3.metric("Rebalance (days)", rebalance_days)
c4.metric("Cash Buffer", f"{cash_buf:.0%}")

with st.expander("Symbols"):
    st.code("\n".join(symbols) if symbols else "(none)")

# --- Load state (created after first bot run) ---
st.subheader("State")
if not state_path.exists():
    st.warning("No state yet. Run the bot once (paper mode) to generate state/state.json.")
    st.stop()

try:
    state = json.loads(state_path.read_text())
except Exception as e:
    st.error(f"Failed to read state: {e}")
    st.stop()

last_plan = state.get("last_plan")
equity_history = state.get("equity_history", [])

if last_plan:
    ts = last_plan.get("ts")
    chosen = last_plan.get("chosen", [])
    weights = last_plan.get("weights", {})
    st.caption(f"Last plan timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts)) if ts else 'N/A'} (UTC)")

    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("**Selected assets**")
        if chosen:
            st.write(", ".join(chosen))
        else:
            st.write("None")

    with c2:
        st.markdown("**Target weights (ex-cash)**")
        if weights:
            dfw = pd.DataFrame(
                [{"Symbol": s, "Weight": float(w)} for s, w in weights.items() if float(w) > 0]
            ).sort_values("Weight", ascending=False)
            st.dataframe(dfw, use_container_width=True)
        else:
            st.write("(no weights)")

else:
    st.info("No last_plan found in state yet.")

# --- Optional: current prices snapshot ---
st.subheader("Price Snapshot (on button)")
if st.button("Fetch latest prices"):
    try:
        ex_name = cfg.get("exchange", {}).get("name", "kraken")
        ex_cls = getattr(ccxt, ex_name.lower())
        client = ex_cls({"enableRateLimit": True, "options": {"adjustForTimeDifference": True}})
        rows = []
        for s in symbols:
            try:
                t = client.fetch_ticker(s)
                px = t.get("last") or t.get("close")
                rows.append({"Symbol": s, "Price": px})
            except Exception as e:
                rows.append({"Symbol": s, "Price": None})
        dfp = pd.DataFrame(rows)
        st.dataframe(dfp, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to fetch prices: {e}")

# --- Optional: equity curve (if you decide to append to equity_history in the bot) ---
if equity_history:
    st.subheader("Equity Curve (from state)")
    try:
        dfe = pd.DataFrame(equity_history, columns=["ts", "equity"])
        dfe["date"] = pd.to_datetime(dfe["ts"], unit="s", utc=True).dt.tz_convert("UTC")
        dfe = dfe.sort_values("date")
        st.line_chart(dfe.set_index("date")["equity"])
    except Exception as e:
        st.warning(f"Could not render equity history: {e}")

st.caption("Tip: Deploy this dashboard on Streamlit Community Cloud and point it to this repo for a free hosted view.")
