
# -*- coding: utf-8 -*-
"""
Laithinho FPL AI V6.3
نسخة عربية متكاملة — إصلاح أخطاء البيانات التاريخية:
- بيانات FPL الحالية
- آخر الجولات
- تاريخ المواسم السابقة
- Home / Away
- سجل اللاعب
- احتمال البداية
- تقييم قرار/مخاطرة القرار (وليس "مخاطرة اللاعب")
- مقارنة لاعبين
- تشكيلة محسنة
- مستشار الكابتن والـChips
- أخبار اللاعب
- واجهة محادثة عربية
- تتضمن الآن طبقة Machine Learning حقيقية مع Backtesting

ملاحظة:
الموسم الحالي 2026/27 بدأ بالفعل؛ GW1 تدخل كبيانات حديثة، بينما المواسم السابقة تُستخدم للتعلم والتوقع.
"""

import html
from io import BytesIO
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pulp
import requests
import streamlit as st
from PIL import Image
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# ============================================================
# إعداد الصفحة
# ============================================================
st.set_page_config(
    page_title="Laithinho FPL AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp{background:#070b10;direction:rtl}
.hero{background:linear-gradient(135deg,#111827,#182231);padding:22px;
border-radius:18px;border:1px solid #00ff87;border-right:6px solid #00ff87;margin-bottom:18px}
.hero h1{color:white;margin:0}.hero p{color:#00ff87;margin:6px 0;font-weight:700}
.pitch{background:linear-gradient(180deg,#087d4f,#14a65f);border:3px solid white;
border-radius:24px;padding:18px;box-shadow:0 10px 35px rgba(0,0,0,.35)}
.player{background:rgba(255,255,255,.97);color:#111;border-radius:12px;padding:8px;
text-align:center;margin:3px;box-shadow:0 3px 8px rgba(0,0,0,.25);min-height:95px}
.shirt{font-size:30px;line-height:32px}.player b{font-size:.88rem}
.capt{background:#ff0055;color:#fff;border-radius:5px;padding:2px 5px;font-weight:800}
.vc{background:#075eff;color:#fff;border-radius:5px;padding:2px 5px;font-weight:800}
.card{background:#121923;border:1px solid #2c3746;border-radius:14px;padding:14px;margin:8px 0}
.ai{background:#111923;border-right:4px solid #00ff87;border-radius:14px;padding:15px;margin:8px 0}
.good{color:#00ff87;font-weight:800}.warn{color:#ffd166;font-weight:800}.bad{color:#ff5c7a;font-weight:800}
.small{font-size:.85rem;color:#aeb8c5}
.pitch-title{text-align:center;color:white;font-weight:900;font-size:1.15rem;margin:4px 0 12px}
.row-label{text-align:center;color:rgba(255,255,255,.85);font-weight:800;margin:5px 0}
.shirt-wrap{display:flex;justify-content:center;align-items:center;height:42px}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>Laithinho FPL AI</h1>
<p>حلّل فريقك مثل FPL: الأداء الحالي + آخر الجولات + التاريخ + المباريات + الأخبار.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# المصادر
# ============================================================
BASE = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0 Laithinho-FPL-AI"}

# بيانات تاريخية عامة من مستودع Vaastav.
# المستودع يضم بيانات GW تاريخية، ومذكور كمصدر للبيانات التاريخية.
HIST_URLS = {
    "2025-26": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/gws/merged_gw.csv",
    "2024-25": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv",
}

@st.cache_data(ttl=1800)
def get_json(path):
    r = requests.get(f"{BASE}/{path}", headers=HEADERS, timeout=25)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=21600)
def get_historical_csv(url):
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return pd.read_csv(pd.io.common.BytesIO(r.content), low_memory=False)

@st.cache_data(ttl=1800)
def load_current():
    b = get_json("bootstrap-static/")
    fx = get_json("fixtures/")
    p = pd.DataFrame(b["elements"])
    teams = pd.DataFrame(b["teams"])
    events = pd.DataFrame(b["events"])
    team_names = teams.set_index("id")["name"].to_dict()

    p["team_name"] = p["team"].map(team_names)
    p["price"] = p["now_cost"] / 10

    numeric = [
        "form","points_per_game","minutes","starts","bonus","bps","ict_index",
        "influence","creativity","threat","selected_by_percent","goals_scored",
        "assists","clean_sheets","expected_goals","expected_assists",
        "expected_goal_involvements","expected_goals_conceded",
        "chance_of_playing_next_round","chance_of_playing_this_round","ep_next"
    ]
    for c in numeric:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0)

    current = events[events["is_current"] == True]
    nxt = events[events["is_next"] == True]
    if not current.empty:
        gw = int(current.iloc[0]["id"])
    elif not nxt.empty:
        gw = int(nxt.iloc[0]["id"])
    else:
        gw = 1
    return p, teams, events, pd.DataFrame(fx), team_names, gw

try:
    df, teams_df, events_df, fixtures, team_names, current_gw = load_current()
except Exception as exc:
    st.error("تعذر تحميل بيانات FPL الحالية.")
    st.code(str(exc))
    st.stop()

# ============================================================
# التاريخ
# ============================================================
@st.cache_data(ttl=21600)
def load_history():
    # تحميل الموسمين بالتوازي لتقليل وقت الإقلاع الأول للتطبيق.
    def fetch(item):
        season, url = item
        try:
            h = get_historical_csv(url)
            h["الموسم"] = season
            return h
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        frames = [x for x in pool.map(fetch, HIST_URLS.items()) if x is not None]

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

with st.spinner("جاري تحميل البيانات التاريخية..."):
    history = load_history()

# ============================================================
# أدوات البيانات
# ============================================================
def num(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)

def normalize(series):
    s = num(series)
    mx = float(s.max())
    return (s / mx).clip(0, 1) if mx > 0 else pd.Series(0, index=s.index)

def upcoming(tid, horizon=6):
    result = []
    for _, x in fixtures.iterrows():
        if pd.isna(x.get("event")):
            continue
        gw = int(x["event"])
        if gw < current_gw or gw > current_gw + horizon:
            continue
        if int(x["team_h"]) == int(tid):
            opp = team_names.get(int(x["team_a"]), "?")
            diff = float(x.get("team_h_difficulty", 3))
            result.append({"gw": gw, "opp": opp, "home": True, "fdr": diff})
        elif int(x["team_a"]) == int(tid):
            opp = team_names.get(int(x["team_h"]), "?")
            diff = float(x.get("team_a_difficulty", 3))
            result.append({"gw": gw, "opp": opp, "home": False, "fdr": diff})
    return result

def fixture_strength(tid, horizon=6):
    fs = upcoming(tid, horizon)
    if not fs:
        return 0.50
    vals = [((5 - f["fdr"]) / 3) + (0.08 if f["home"] else 0) for f in fs]
    return float(np.clip(np.mean(vals), .05, 1.25))

def dgw_count(tid, horizon=8):
    fs = upcoming(tid, horizon)
    counts = {}
    for f in fs:
        counts[f["gw"]] = counts.get(f["gw"], 0) + 1
    return sum(v >= 2 for v in counts.values())

# ============================================================
# V6.3 — محرك المباراة القادمة
# ============================================================
# الفكرة الأساسية: قبل أن نتوقع نقاط اللاعب، نتوقع المباراة نفسها.
# نستخدم قوة الفريق الرسمية في FPL + Home/Away + FDR، ثم نحوّلها إلى
# تقدير أهداف متوقعة واحتمالات فوز/تعادل/خسارة وClean Sheet.
# هذه طبقة مستقلة عن ML حتى لا يصبح النموذج أسير متوسطات اللاعب.

def _team_strength_row(tid):
    z = teams_df[teams_df["id"] == int(tid)]
    return z.iloc[0] if not z.empty else pd.Series(dtype=object)

def _strength(row, key, fallback=1000.0):
    try:
        v = float(row.get(key, fallback))
        return v if np.isfinite(v) and v > 0 else fallback
    except Exception:
        return fallback

@st.cache_data(ttl=1800)
def build_match_predictions(_fixtures, _teams_df, _team_names, gw):
    # المتوسطات الافتراضية للدوري تُستخدم فقط كمرساة، ثم نعدلها بقوة الفرق.
    league_goal = 1.45
    rows = []
    future = _fixtures[_fixtures["event"].notna()].copy() if "event" in _fixtures.columns else pd.DataFrame()
    if future.empty:
        return pd.DataFrame()
    future["event"] = pd.to_numeric(future["event"], errors="coerce")
    future = future[future["event"] >= int(gw)].copy()
    if future.empty:
        return pd.DataFrame()

    for _, fx in future.sort_values("event").iterrows():
        th, ta = int(fx["team_h"]), int(fx["team_a"])
        rh, ra = _team_strength_row(th), _team_strength_row(ta)
        # FPL strength_* fields are scaled ratings. We use the correct home/away
        # attack and defence fields when available.
        ah = _strength(rh, "strength_attack_home", _strength(rh, "strength", 1000))
        da = _strength(rh, "strength_defence_home", _strength(rh, "strength", 1000))
        aa = _strength(ra, "strength_attack_away", _strength(ra, "strength", 1000))
        dh = _strength(ra, "strength_defence_away", _strength(ra, "strength", 1000))

        # Convert ratings into relative attack/defence multipliers. Clamp to avoid
        # absurd early-season predictions when only one GW exists.
        att_h = np.clip(ah / 1100.0, .55, 1.75)
        att_a = np.clip(aa / 1100.0, .55, 1.75)
        def_h = np.clip(da / 1100.0, .55, 1.75)
        def_a = np.clip(dh / 1100.0, .55, 1.75)

        xg_h = league_goal * att_h * (2.0 / (1.0 + def_a)) * 1.08
        xg_a = league_goal * att_a * (2.0 / (1.0 + def_h)) * .92
        xg_h = float(np.clip(xg_h, .20, 3.80))
        xg_a = float(np.clip(xg_a, .15, 3.20))

        # FDR is an explicit opponent-strength signal.
        fdr_h = float(fx.get("team_h_difficulty", 3) or 3)
        fdr_a = float(fx.get("team_a_difficulty", 3) or 3)
        xg_h *= float(np.clip(1.08 - .055 * (fdr_h - 3), .82, 1.18))
        xg_a *= float(np.clip(1.08 - .055 * (fdr_a - 3), .82, 1.18))

        total = xg_h + xg_a
        p_h_win = float(np.clip(.50 + .16 * (xg_h - xg_a), .08, .90))
        p_a_win = float(np.clip(.50 + .16 * (xg_a - xg_h), .08, .90))
        p_draw = float(np.clip(1.0 - p_h_win - p_a_win + .18, .08, .38))
        norm = p_h_win + p_a_win + p_draw
        p_h_win, p_a_win, p_draw = p_h_win/norm, p_a_win/norm, p_draw/norm

        cs_h = float(np.exp(-xg_a))
        cs_a = float(np.exp(-xg_h))
        over25 = float(1 - np.exp(-total) * (1 + total + (total**2)/2))
        h_2plus = float(1 - np.exp(-xg_h) * (1 + xg_h))
        a_2plus = float(1 - np.exp(-xg_a) * (1 + xg_a))

        rows.append({
            "gw": int(fx["event"]), "fixture_id": int(fx.get("id", -1)),
            "home_id": th, "away_id": ta,
            "home": _team_names.get(th, "?"), "away": _team_names.get(ta, "?"),
            "xg_home": xg_h, "xg_away": xg_a,
            "home_win": p_h_win, "draw": p_draw, "away_win": p_a_win,
            "cs_home": cs_h, "cs_away": cs_a,
            "home_2plus": h_2plus, "away_2plus": a_2plus,
            "over_2_5": over25,
            "fdr_home": fdr_h, "fdr_away": fdr_a,
        })
    return pd.DataFrame(rows)

match_predictions = build_match_predictions(fixtures, teams_df, team_names, current_gw)

def next_match(tid):
    if match_predictions.empty:
        return None
    x = match_predictions[(match_predictions["home_id"] == int(tid)) | (match_predictions["away_id"] == int(tid))]
    return x.sort_values("gw").iloc[0] if not x.empty else None

def fixture_label_for_player(row):
    m = next_match(int(row["team"]))
    if m is None:
        return pd.Series({
            "المباراة_القادمة": "غير متوفرة", "المباراة_القادمة_المكان": "—",
            "أهداف_الفريق_المتوقعة": 0.0, "احتمال_الفوز": 0.0,
            "احتمال_كلين_شيت": 0.0, "احتمال_الفريق_يسجل_هدفين": 0.0,
            "قوة_المواجهة": .5, "ملخص_المباراة": "لا توجد مباراة قادمة متاحة."
        })
    home_side = int(m["home_id"]) == int(row["team"])
    opp = m["away"] if home_side else m["home"]
    team_xg = m["xg_home"] if home_side else m["xg_away"]
    win = m["home_win"] if home_side else m["away_win"]
    cs = m["cs_home"] if home_side else m["cs_away"]
    two = m["home_2plus"] if home_side else m["away_2plus"]
    fdr = m["fdr_home"] if home_side else m["fdr_away"]
    venue = "داخل الأرض 🏠" if home_side else "خارج الأرض ✈️"
    # Higher is better: easy FDR + home boost + expected team goals.
    match_score = float(np.clip(.55 * ((5-fdr)/4) + .25 * min(team_xg/2.2,1) + .20 * (1 if home_side else .75), 0, 1))
    return pd.Series({
        "المباراة_القادمة": f"{row['team_name']} × {opp}",
        "المباراة_القادمة_المكان": venue,
        "أهداف_الفريق_المتوقعة": team_xg,
        "احتمال_الفوز": win, "احتمال_كلين_شيت": cs,
        "احتمال_الفريق_يسجل_هدفين": two, "قوة_المواجهة": match_score,
        "ملخص_المباراة": f"{row['team_name']} ضد {opp}: xG الفريق {team_xg:.2f}، فوز {win*100:.0f}%، Clean Sheet {cs*100:.0f}%، {venue}."
    })

match_cols = df.apply(fixture_label_for_player, axis=1)
df = pd.concat([df, match_cols], axis=1)

# ============================================================
# توقع نقاط قائم على المباراة نفسها
# ============================================================
def _team_xg_sum(tid):
    m = match_predictions[(match_predictions["home_id"] == int(tid)) | (match_predictions["away_id"] == int(tid))] if not match_predictions.empty else pd.DataFrame()
    if m.empty:
        return 1.45
    z = m.iloc[0]
    return float(z["xg_home"] if int(z["home_id"]) == int(tid) else z["xg_away"])

# توزيع xG الفردي من بيانات FPL الحالية؛ إذا لم توجد xG، نستخدم xGI كنقطة
# انطلاق حتى لا نكسر اللاعبين الجدد.
def add_match_aware_player_projection(frame):
    out = frame.copy()
    team_xg_map = {}
    team_xg_total = {}
    for tid in out["team"].dropna().unique():
        team_xg_map[int(tid)] = _team_xg_sum(int(tid))
        g = out[out["team"] == tid]
        team_xg_total[int(tid)] = float(num(g.get("expected_goals", 0)).sum())

    exp_goal = num(out.get("expected_goals", 0))
    exp_assist = num(out.get("expected_assists", 0))
    exp_xgi = num(out.get("expected_goal_involvements", 0))
    team_share = []
    for idx, r in out.iterrows():
        tid = int(r["team"])
        denom = team_xg_total.get(tid, 0.0)
        if denom > 0 and float(exp_goal.loc[idx]) > 0:
            share = float(exp_goal.loc[idx] / denom)
        else:
            g = out[out["team"] == tid]
            xgi_total = float(num(g.get("expected_goal_involvements", 0)).sum())
            share = float(exp_xgi.loc[idx] / xgi_total) if xgi_total > 0 else 0.05
        team_share.append(float(np.clip(share, 0.01, .45)))
    out["حصة_أهداف_الفريق"] = team_share
    out["xG_الجولة_القادمة"] = [team_xg_map.get(int(t), 1.45) * s for t, s in zip(out["team"], out["حصة_أهداف_الفريق"])]
    out["xA_الجولة_القادمة"] = np.clip(exp_assist * (0.7 + 0.6 * out["قوة_المواجهة"]), 0, 1.2)

    pos = num(out["element_type"])
    minutes_factor = out["احتمال_البداية_والتواجد"].clip(0,1)
    cs = out["احتمال_كلين_شيت"].clip(0,1)
    # FPL scoring expectation approximation: appearance + goals + assists + CS +
    # a small bonus component. This is intentionally an interpretable layer.
    appearance = 2.0 * minutes_factor
    goal_points = np.where(pos == 4, 4.0, np.where(pos == 3, 5.0, np.where(pos == 2, 6.0, 10.0))) * out["xG_الجولة_القادمة"]
    assist_points = 3.0 * out["xA_الجولة_القادمة"]
    cs_points = np.where(pos == 2, 4.0, np.where(pos == 3, 1.0, 0.0)) * cs * minutes_factor
    bonus = .35 * out["مؤشر_الاعتماد_الأساسي"] + .25 * out["bps_n"] + .20 * out["ict_n"]
    match_raw = appearance + goal_points + assist_points + cs_points + bonus

    # Match-aware blend: the fixture can move the prediction materially, but ML
    # and historical/current form remain in the decision.
    out["توقع_المباراة"] = np.clip(match_raw, .2, 18)
    out["النقاط_المتوقعة_المباراة_الواعية"] = np.clip(
        .55 * out["توقع_المباراة"] + .30 * out["النقاط_المتوقعة"] + .15 * out["توقع_ML"].fillna(out["النقاط_المتوقعة"]),
        .2, 18
    )
    # Strong match signals can lift ceiling; poor fixtures can lower it.
    out["سقف_النقاط_التقديري"] = np.clip(
        out["النقاط_المتوقعة_المباراة_الواعية"] + 4.0 * out["xG_الجولة_القادمة"] + 2.0 * out["xA_الجولة_القادمة"] + 2.0 * out["احتمال_كلين_شيت"] * (pos == 2),
        .5, 25
    )
    return out

# ملاحظة: لا نستدعي طبقة المباراة هنا، لأن أعمدة التواجد واعتماد المدرب
# وميزات ML تُبنى لاحقًا. الاستدعاء المبكر كان سبب KeyError في Streamlit.

# ============================================================
# التاريخ لكل لاعب
# ============================================================
def historical_player_stats(name):
    if history.empty or "name" not in history.columns:
        return {
            "hist_points": 0, "hist_ppg": 0, "hist_minutes": 0,
            "hist_starts": 0, "hist_xgi": 0, "hist_home": 0,
            "hist_away": 0, "hist_gws": 0
        }

    h = history[history["name"].astype(str).str.lower() == str(name).lower()].copy()
    if h.empty:
        # fallback جزئي: الاسم يحتوي على الاسم الحالي
        h = history[history["name"].astype(str).str.contains(str(name), case=False, na=False)].copy()

    if h.empty:
        return {
            "hist_points": 0, "hist_ppg": 0, "hist_minutes": 0,
            "hist_starts": 0, "hist_xgi": 0, "hist_home": 0,
            "hist_away": 0, "hist_gws": 0
        }

    points_col = "total_points" if "total_points" in h else "total_points"
    minutes_col = "minutes" if "minutes" in h else None
    xgi_col = "expected_goal_involvements" if "expected_goal_involvements" in h else None

    points = num(h[points_col]) if points_col in h else pd.Series(0, index=h.index)
    minutes = num(h[minutes_col]) if minutes_col else pd.Series(0, index=h.index)
    xgi = num(h[xgi_col]) if xgi_col else pd.Series(0, index=h.index)

    # في ملفات merged_gw، home/away عادة يمكن استنتاجه من was_home.
    home_mask = h["was_home"].astype(bool) if "was_home" in h.columns else pd.Series(False, index=h.index)

    return {
        "hist_points": float(points.sum()),
        "hist_ppg": float(points.mean()) if len(points) else 0,
        "hist_minutes": float(minutes.sum()),
        "hist_starts": float((minutes >= 60).sum()),
        "hist_xgi": float(xgi.sum()),
        "hist_home": float(points[home_mask].mean()) if home_mask.any() else 0,
        "hist_away": float(points[~home_mask].mean()) if (~home_mask).any() else 0,
        "hist_gws": int(len(h))
    }

# نبني الإحصاءات التاريخية مرة واحدة للاعبين الموجودين حاليًا.
@st.cache_data(ttl=21600)
def build_history_summary(current_names):
    # مهم: web_name قد يتكرر بين أكثر من لاعب، وSeries.map()
    # يسبب InvalidIndexError إذا كان فهرس المصدر يحتوي أسماء مكررة.
    # لذلك نبني قاموسًا بمفتاح فريد لكل اسم.
    unique_names = sorted({
        str(x).strip() for x in current_names
        if pd.notna(x) and str(x).strip()
    })

    rows = []
    for name in unique_names:
        s = historical_player_stats(name)
        s["name"] = name
        rows.append(s)

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)

    # حماية إضافية من أي تكرار غير متوقع في الاسم.
    summary = summary.drop_duplicates(subset=["name"], keep="first")
    return summary.set_index("name")

history_summary = build_history_summary(tuple(df["web_name"].fillna("").tolist()))

# نستخدم merge بدل Series.map حتى لا يحدث InvalidIndexError
# وحتى يبقى لكل صف في جدول اللاعبين الحالي صفه الأصلي.
if not history_summary.empty:
    hist_reset = history_summary.reset_index()
    df = df.merge(
        hist_reset,
        how="left",
        left_on="web_name",
        right_on="name",
        suffixes=("", "_history")
    )
    if "name" in df.columns:
        df = df.drop(columns=["name"])

for c in [
    "hist_points", "hist_ppg", "hist_minutes", "hist_starts",
    "hist_xgi", "hist_home", "hist_away", "hist_gws"
]:
    if c not in df.columns:
        df[c] = 0
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# ============================================================
# آخر الجولات
# ============================================================
@st.cache_data(ttl=21600)
def recent_stats():
    if history.empty or "name" not in history.columns or "GW" not in history.columns:
        return pd.DataFrame()

    h = history.copy()
    h["GW"] = num(h["GW"])
    h["total_points"] = num(h.get("total_points", 0))
    h["minutes"] = num(h.get("minutes", 0))
    h["expected_goal_involvements"] = num(h.get("expected_goal_involvements", 0))

    rows = []
    for name, g in h.groupby("name"):
        g = g.sort_values(["الموسم", "GW"])
        last5 = g.tail(5)
        rows.append({
            "name": name,
            "آخر5_نقاط": last5["total_points"].sum(),
            "آخر5_متوسط": last5["total_points"].mean(),
            "آخر5_دقائق": last5["minutes"].sum(),
            "آخر5_دقائق_متوسط": last5["minutes"].mean(),
            "آخر5_بدايات": float((last5["minutes"] >= 60).sum()),
            "آخر5_xGI": last5["expected_goal_involvements"].sum(),
            "آخر10_متوسط": g.tail(10)["total_points"].mean(),
        })
    return pd.DataFrame(rows).set_index("name")

recent = recent_stats()
if not recent.empty:
    recent_reset = recent.reset_index().drop_duplicates(subset=["name"], keep="last")
    df = df.merge(
        recent_reset,
        how="left",
        left_on="web_name",
        right_on="name",
        suffixes=("", "_recent")
    )
    if "name" in df.columns:
        df = df.drop(columns=["name"])
    for c in ["آخر5_نقاط","آخر5_متوسط","آخر5_دقائق","آخر5_دقائق_متوسط","آخر5_بدايات","آخر5_xGI","آخر10_متوسط"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
else:
    for c in ["آخر5_نقاط","آخر5_متوسط","آخر5_دقائق","آخر5_دقائق_متوسط","آخر5_بدايات","آخر5_xGI","آخر10_متوسط"]:
        df[c] = 0

# ============================================================
# نموذج النقاط الحالي
# ============================================================
df["form_n"] = normalize(df["form"])
df["ppg_n"] = normalize(df["points_per_game"])
df["xgi_n"] = normalize(df["expected_goal_involvements"])
df["ict_n"] = normalize(df["ict_index"])
df["bps_n"] = normalize(df["bps"])
df["ep_n"] = normalize(df["ep_next"])
df["hist_ppg_n"] = normalize(df["hist_ppg"])
df["hist_xgi_n"] = normalize(df["hist_xgi"])
df["recent5_n"] = normalize(df["آخر5_متوسط"])
df["recent10_n"] = normalize(df["آخر10_متوسط"])

df["minutes_n"] = (df["minutes"] / max(float(df["minutes"].max()), 1)).clip(0, 1)
df["starts_n"] = (df["starts"] / max(float(df["starts"].max()), 1)).clip(0, 1)

# مهم جدًا: لا نحول 0% إلى 100%.
# في بيانات FPL، الصفر قد يعني أن اللاعب غير متاح فعلًا.
# نعامل القيمة المفقودة فقط كـ"غير مُقيّمة"، ونستفيد من status لتثبيت القرار.
def availability_profile(row):
    status = str(row.get("status", "")).lower()
    raw_next = pd.to_numeric(pd.Series([row.get("chance_of_playing_next_round")]), errors="coerce").iloc[0]
    raw_this = pd.to_numeric(pd.Series([row.get("chance_of_playing_this_round")]), errors="coerce").iloc[0]

    next_pct = 100.0 if pd.isna(raw_next) and status == "a" else (0.0 if pd.isna(raw_next) else float(np.clip(raw_next, 0, 100)))
    this_pct = 100.0 if pd.isna(raw_this) and status == "a" else (0.0 if pd.isna(raw_this) else float(np.clip(raw_this, 0, 100)))

    if status in {"i", "s", "u"} or next_pct <= 0:
        label = "🔴 غير متاح"
        severity = 1.0
    elif status == "d" or next_pct < 75:
        label = "🟡 مشكوك بمشاركته"
        severity = 0.55
    else:
        label = "🟢 متاح مبدئيًا"
        severity = 0.0

    news_text = str(row.get("news", "") or "").lower()
    long_terms = [
        "long term", "long-term", "for months", "months", "surgery",
        "out until", "out for", "several weeks", "weeks", "weeks out",
        "set to miss", "ruled out for the season"
    ]
    long_absence = (status in {"i", "u"} and next_pct <= 25) or any(k in news_text for k in long_terms)

    return pd.Series({
        "نسبة_التواجد_القادمة": next_pct / 100.0,
        "نسبة_التواجد_الحالية": this_pct / 100.0,
        "احتمال_البداية_والتواجد": np.clip((next_pct * .70 + this_pct * .30) / 100.0, 0, 1),
        "حالة_التوفر": label,
        "غياب_طويل_أو_غير_محدد": bool(long_absence),
        "شدة_الغياب": severity,
    })

avail = df.apply(availability_profile, axis=1)
df = pd.concat([df, avail], axis=1)

# ============================================================
# اعتماد المدرب على اللاعب / احتمالية أن يكون أساسيًا
# ============================================================
# لا نساوي بين لاعب يملك نقاطًا جيدة لكنه حبيس الدكة، ولاعب يعتمد عليه
# المدرب ويبدأ معظم المباريات. في بداية الموسم نستخدم GW المتاحة فقط،
# ثم تزداد دقة المؤشر تلقائيًا مع مرور الجولات.
completed_gws = max(int(current_gw) - 1, 1)
df["معدل_البدايات_الحالي"] = (df["starts"] / completed_gws).clip(0, 1)
df["معدل_الدقائق_الحالي"] = (df["minutes"] / (completed_gws * 90.0)).clip(0, 1)
df["معدل_البدايات_التاريخي"] = (df["hist_starts"] / df["hist_gws"].replace(0, np.nan)).fillna(0).clip(0, 1)
df["معدل_الدقائق_التاريخي"] = (df["hist_minutes"] / (df["hist_gws"].replace(0, np.nan) * 90.0)).fillna(0).clip(0, 1)

# عند توفر بيانات الموسم الحالي نعطيها الوزن الأكبر. للاعب جديد/منقول
# بلا دقائق حالية، نستفيد من سجله التاريخي بدل اعتباره لاعب دكة تلقائيًا.
df["مؤشر_الاعتماد_الأساسي"] = np.where(
    (df["minutes"] + df["starts"]) > 0,
    .55 * df["معدل_البدايات_الحالي"] + .25 * df["معدل_الدقائق_الحالي"] + .20 * df["معدل_البدايات_التاريخي"],
    .70 * df["معدل_البدايات_التاريخي"] + .30 * df["معدل_الدقائق_التاريخي"]
).clip(0, 1)

# إشارات إضافية من آخر 5 جولات تاريخية عندما تكون متاحة.
recent_start_rate = (df["آخر5_بدايات"] / 5.0).clip(0, 1)
recent_min_rate = (df["آخر5_دقائق_متوسط"] / 90.0).clip(0, 1)
df["مؤشر_الاعتماد_الأساسي"] = np.where(
    (df["آخر5_بدايات"] + df["آخر5_دقائق_متوسط"]) > 0,
    .55 * df["مؤشر_الاعتماد_الأساسي"] + .30 * recent_start_rate + .15 * recent_min_rate,
    df["مؤشر_الاعتماد_الأساسي"]
).clip(0, 1)

df["قوة_المباريات"] = df["team"].apply(lambda tid: fixture_strength(tid))
df["عدد_DGW"] = df["team"].apply(lambda tid: dgw_count(tid))

# لا نسمح للجولة الماضية أن تهيمن.
df["الأساس"] = (
    .16 * df["form_n"] +
    .13 * df["ppg_n"] +
    .16 * df["xgi_n"] +
    .08 * df["ict_n"] +
    .05 * df["bps_n"] +
    .10 * df["minutes_n"] +
    .08 * df["starts_n"] +
    .09 * df["recent5_n"] +
    .05 * df["recent10_n"] +
    .06 * df["hist_ppg_n"] +
    .04 * df["hist_xgi_n"] +
    .07 * df["احتمال_البداية_والتواجد"] +
    .10 * df["مؤشر_الاعتماد_الأساسي"]
)

df["النقاط_المتوقعة"] = (
    11.0 * (
        .72 * df["الأساس"] +
        .18 * df["ep_n"] +
        .10 * df["قوة_المباريات"].clip(0,1)
    )
).clip(.2, 13)

# availability gate: اللاعب غير المتاح أو الغائب طويلًا لا يجوز أن
# ينافس اللاعب المتاح في الاختيار النهائي. لا نحذف اللاعب من التحليل،
# بل نُبقيه ظاهرًا مع عقوبة قوية على التوقع.
df["النقاط_المتوقعة"] *= (
    .05 + .95 * df["احتمال_البداية_والتواجد"].clip(0,1)
)
df.loc[df["غياب_طويل_أو_غير_محدد"], "النقاط_المتوقعة"] *= 0.15

# DGW bonus محدود ومشروط بالمشاركة.
df["النقاط_المتوقعة"] += df["عدد_DGW"].clip(0,2) * (
    .35 * df["احتمال_البداية_والتواجد"] + .15 * df["xgi_n"]
)

# ============================================================
# "مخاطرة القرار" وليس مخاطرة اللاعب
# ============================================================
# Decision Risk = عدم اليقين الذي يواجه قرارك الآن:
# rotation + availability + small sample + fixture volatility + dependence on recent form.
df["مخاطرة_القرار"] = np.clip(
    100 * (
        .32 * (1 - df["احتمال_البداية_والتواجد"]) +
        .12 * (1 - df["starts_n"]) +
        .12 * (1 - df["minutes_n"]) +
        .10 * (1 - df["مؤشر_الاعتماد_الأساسي"]) +
        .14 * (1 - df["hist_ppg_n"]) +
        .14 * (1 - df["قوة_المباريات"].clip(0,1)) +
        .20 * df["شدة_الغياب"]
    ),
    0, 100
)

def risk_label(x):
    if x >= 65:
        return "🔴 مخاطرة قرار مرتفعة"
    if x >= 40:
        return "🟡 مخاطرة قرار متوسطة"
    return "🟢 مخاطرة قرار منخفضة"


# ============================================================
# V6.2 — Machine Learning Prediction Engine
# ============================================================
# هذا نموذج ML فعلي: يتعلم من صفوف تاريخية "قبل الجولة" ويتنبأ
# بالنقاط الفعلية للجولة التالية. لا نستخدم نقاط نفس الجولة كمدخل.
ML_FEATURES = [
    "ml_prev_points", "ml_roll3_points", "ml_roll5_points",
    "ml_prev_minutes", "ml_roll5_minutes", "ml_prev_starts",
    "ml_prev_goals", "ml_prev_assists", "ml_prev_xgi",
    "ml_prev_bonus", "ml_prev_bps", "ml_prev_ict",
    "ml_season_ppg", "ml_price", "ml_position", "ml_home",
    "ml_start_rate", "ml_minutes_rate",
]

def _first_existing(frame, names, default=0):
    for n in names:
        if n in frame.columns:
            return pd.to_numeric(frame[n], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)

@st.cache_data(ttl=21600)
def build_ml_training_data(history_df):
    if history_df is None or history_df.empty:
        return pd.DataFrame()

    h = history_df.copy()
    required = {"name", "GW", "total_points"}
    if not required.issubset(h.columns):
        return pd.DataFrame()

    h["GW"] = pd.to_numeric(h["GW"], errors="coerce")
    h["total_points"] = pd.to_numeric(h["total_points"], errors="coerce").fillna(0)
    h = h.dropna(subset=["name", "GW"]).copy()
    h["name"] = h["name"].astype(str).str.strip()

    # season + player + GW = chronological order.
    sort_cols = [c for c in ["name", "الموسم", "GW"] if c in h.columns]
    h = h.sort_values(sort_cols).copy()

    numeric_sources = {
        "minutes": ["minutes"],
        "starts": ["starts"],
        "goals": ["goals_scored"],
        "assists": ["assists"],
        "xgi": ["expected_goal_involvements"],
        "bonus": ["bonus"],
        "bps": ["bps"],
        "ict": ["ict_index"],
        "ppg": ["points_per_game"],
        "price": ["value", "now_cost"],
        "position": ["element_type"],
        "home": ["was_home"],
    }

    for key, names in numeric_sources.items():
        h[f"__{key}"] = _first_existing(h, names)

    g = h.groupby(["name"] + (["الموسم"] if "الموسم" in h.columns else []), sort=False)

    # All rolling features are shifted first: no future leakage.
    for key in ["minutes", "starts", "goals", "assists", "xgi", "bonus", "bps", "ict"]:
        prev = g[f"__{key}"].shift(1)
        h[f"ml_prev_{key}"] = prev
        group_series = [h["name"]] + ([h["الموسم"]] if "الموسم" in h.columns else [])
        h[f"ml_roll5_{key}"] = prev.groupby(group_series, sort=False).transform(
            lambda s: s.rolling(5, min_periods=1).mean()
        )

    prev_points = g["total_points"].shift(1)
    h["ml_prev_points"] = prev_points
    group_keys = ["name"] + (["الموسم"] if "الموسم" in h.columns else [])
    h["ml_roll3_points"] = prev_points.groupby(
        [h[k] for k in group_keys], sort=False
    ).transform(lambda s: s.rolling(3, min_periods=1).mean())
    h["ml_roll5_points"] = prev_points.groupby(
        [h[k] for k in group_keys], sort=False
    ).transform(lambda s: s.rolling(5, min_periods=1).mean())
    h["ml_roll5_minutes"] = h["ml_prev_minutes"].groupby(
        [h[k] for k in group_keys], sort=False
    ).transform(lambda s: s.rolling(5, min_periods=1).mean())

    # معدل البداية/الدقائق قبل المباراة: أهم إشارة على اعتماد المدرب.
    h["ml_start_rate"] = (h["ml_prev_starts"] / h["GW"].clip(lower=1)).clip(0, 1)
    h["ml_minutes_rate"] = (h["ml_prev_minutes"] / (h["GW"].clip(lower=1) * 90.0)).clip(0, 1)

    h["ml_season_ppg"] = h["__ppg"].shift(1).fillna(0)
    h["ml_price"] = h["__price"].shift(1).fillna(0)
    h["ml_position"] = h["__position"].fillna(0)
    h["ml_home"] = h["__home"].fillna(0).astype(float)

    # Remove rows where we have no previous information at all.
    h = h[h["ml_prev_points"].notna()].copy()
    h = h.replace([np.inf, -np.inf], np.nan)
    return h[ML_FEATURES + ["total_points", "الموسم", "GW", "name"]].copy()

@st.cache_resource(ttl=21600)
def train_ml_model(history_df):
    train = build_ml_training_data(history_df)
    if train.empty or len(train) < 200:
        return None, {"trained": False, "rows": len(train)}

    # Time-aware validation: train on the older season, validate on the newer one.
    seasons = sorted(train["الموسم"].dropna().unique()) if "الموسم" in train.columns else []
    if len(seasons) >= 2:
        train_part = train[train["الموسم"] == seasons[0]].copy()
        valid_part = train[train["الموسم"] == seasons[-1]].copy()
    else:
        cutoff = train["GW"].quantile(.8)
        train_part = train[train["GW"] <= cutoff].copy()
        valid_part = train[train["GW"] > cutoff].copy()

    X_train = train_part[ML_FEATURES]
    y_train = train_part["total_points"]
    X_valid = valid_part[ML_FEATURES]
    y_valid = valid_part["total_points"]

    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_valid_i = imputer.transform(X_valid)

    model = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.045,
        max_leaf_nodes=24,
        l2_regularization=1.5,
        random_state=42,
    )
    model.fit(X_train_i, y_train)

    mae = mean_absolute_error(y_valid, model.predict(X_valid_i)) if len(valid_part) else None

    # Final production model learns from ALL historical rows.
    X_all = imputer.fit_transform(train[ML_FEATURES])
    final_model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=24,
        l2_regularization=1.5,
        random_state=42,
    )
    final_model.fit(X_all, train["total_points"])

    return (final_model, imputer), {
        "trained": True,
        "rows": len(train),
        "validation_rows": len(valid_part),
        "mae": float(mae) if mae is not None else None,
        "seasons": [str(s) for s in seasons],
    }

# Current-season features. With only GW1 available, the model gets GW1 as
# recent information and the historical columns above provide the longer memory.
def make_current_ml_features(frame):
    out = pd.DataFrame(index=frame.index)
    out["ml_prev_points"] = num(frame.get("total_points", 0))
    out["ml_roll3_points"] = num(frame.get("total_points", 0))
    out["ml_roll5_points"] = num(frame.get("total_points", 0))
    out["ml_prev_minutes"] = num(frame.get("minutes", 0))
    out["ml_roll5_minutes"] = num(frame.get("minutes", 0))
    out["ml_prev_starts"] = num(frame.get("starts", 0))
    out["ml_prev_goals"] = num(frame.get("goals_scored", 0))
    out["ml_prev_assists"] = num(frame.get("assists", 0))
    out["ml_prev_xgi"] = num(frame.get("expected_goal_involvements", 0))
    out["ml_prev_bonus"] = num(frame.get("bonus", 0))
    out["ml_prev_bps"] = num(frame.get("bps", 0))
    out["ml_prev_ict"] = num(frame.get("ict_index", 0))
    out["ml_season_ppg"] = num(frame.get("points_per_game", 0))
    out["ml_price"] = num(frame.get("price", 0))
    out["ml_position"] = num(frame.get("element_type", 0))
    out["ml_home"] = 0.0
    completed = max(int(current_gw) - 1, 1)
    out["ml_start_rate"] = (num(frame.get("starts", 0)) / completed).clip(0, 1)
    out["ml_minutes_rate"] = (num(frame.get("minutes", 0)) / (completed * 90.0)).clip(0, 1)

    # Next fixture: home/away is known before kickoff and is safe to use.
    for i, (_, row) in enumerate(frame.iterrows()):
        fs = upcoming(row["team"], horizon=1)
        if fs:
            out.loc[row.name, "ml_home"] = 1.0 if fs[0]["home"] else 0.0
    return out[ML_FEATURES]

with st.spinner("جاري تدريب محرك التوقعات ML على المواسم السابقة..."):
    ml_bundle, ml_info = train_ml_model(history)

if ml_bundle is not None:
    ml_model, ml_imputer = ml_bundle
    ml_features_now = make_current_ml_features(df)
    ml_pred = ml_model.predict(ml_imputer.transform(ml_features_now))
    # Blend ML with the deterministic engine rather than blindly trusting one model.
    df["توقع_ML"] = np.clip(ml_pred, 0.0, 15.0)
    df["النقاط_المتوقعة_قبل_ML"] = df["النقاط_المتوقعة"]
    df["النقاط_المتوقعة"] = (
        0.65 * df["النقاط_المتوقعة_قبل_ML"] +
        0.35 * df["توقع_ML"]
    ).clip(.2, 15)

    # طبقة التوفر تطبق مرة أخيرة بعد دمج ML حتى لا يستطيع النموذج
    # رفع لاعب مصاب/غائب طويلًا إلى ترتيب مضلل.
    df["النقاط_المتوقعة"] *= (
        .05 + .95 * df["احتمال_البداية_والتواجد"].clip(0,1)
    )
    # اعتماد المدرب يدخل في التوقع النهائي، لكن بشكل تدريجي حتى لا نعاقب
    # لاعبًا جديدًا بعدد جولات قليل أو انتقالًا حديثًا.
    starter_factor = (0.72 + 0.28 * df["مؤشر_الاعتماد_الأساسي"].clip(0,1))
    df["النقاط_المتوقعة"] *= starter_factor
    df.loc[df["غياب_طويل_أو_غير_محدد"], "النقاط_المتوقعة"] *= 0.15
else:
    df["توقع_ML"] = np.nan
    ml_info = {"trained": False, "rows": 0, "mae": None, "seasons": []}

# ============================================================
# الدمج النهائي: المباراة + التاريخ + ML + التواجد
# ============================================================
# يجب أن يأتي بعد بناء جميع الأعمدة التي تحتاجها طبقة المباراة.
df = add_match_aware_player_projection(df)
df["النقاط_المتوقعة"] = df["النقاط_المتوقعة_المباراة_الواعية"]

# حماية نهائية من المصاب/الغائب طويلًا ومن اللاعب غير المتاح.
df["النقاط_المتوقعة"] *= (
    .05 + .95 * df["احتمال_البداية_والتواجد"].clip(0, 1)
)
df["النقاط_المتوقعة"] *= (
    .72 + .28 * df["مؤشر_الاعتماد_الأساسي"].clip(0, 1)
)
df.loc[df["غياب_طويل_أو_غير_محدد"], "النقاط_المتوقعة"] *= 0.15
df["النقاط_المتوقعة"] = df["النقاط_المتوقعة"].clip(.2, 18)

# ============================================================
# تحسين الفريق
# ============================================================
def optimize_squad(budget):
    # التشكيلة المثالية لا تختار لاعبًا غير متاح طويلًا.
    # اللاعب المشكوك بمشاركته قد يدخل إذا بقيت قيمته قوية، لكن اللاعب
    # المصاب/الموقوف/غير المتاح 0% لا يدخل في الـ15 المثالية.
    d = df[(df["price"] > 0)].copy()
    d = d[~((d["حالة_التوفر"] == "🔴 غير متاح") | (d["غياب_طويل_أو_غير_محدد"] == True))]
    d = d.sort_values("النقاط_المتوقعة", ascending=False).head(450)
    ids = d.index.tolist()

    prob = pulp.LpProblem("Laithinho", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("اختيار", ids, cat="Binary")

    prob += pulp.lpSum(d.loc[i, "النقاط_المتوقعة"] * x[i] for i in ids)
    prob += pulp.lpSum(x[i] for i in ids) == 15
    prob += pulp.lpSum(d.loc[i, "price"] * x[i] for i in ids) <= budget

    for typ, n in [(1,2),(2,5),(3,5),(4,3)]:
        prob += pulp.lpSum(x[i] for i in ids if int(d.loc[i,"element_type"]) == typ) == n

    for tid in d["team"].unique():
        prob += pulp.lpSum(x[i] for i in ids if d.loc[i,"team"] == tid) <= 3

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return pd.DataFrame()

    return d.loc[[i for i in ids if x[i].value() == 1]].copy()

def best_xi(squad):
    best = pd.DataFrame()
    best_score = -1
    for dn, mn, fn in [(3,4,3),(3,5,2),(4,4,2),(4,3,3),(5,4,1),(5,3,2)]:
        de = squad[squad.element_type == 2].nlargest(dn, "النقاط_المتوقعة")
        mi = squad[squad.element_type == 3].nlargest(mn, "النقاط_المتوقعة")
        fw = squad[squad.element_type == 4].nlargest(fn, "النقاط_المتوقعة")
        gk = squad[squad.element_type == 1].nlargest(1, "النقاط_المتوقعة")
        if len(de) == dn and len(mi) == mn and len(fw) == fn:
            z = pd.concat([gk,de,mi,fw])
            score = z["النقاط_المتوقعة"].sum()
            if score > best_score:
                best_score = score
                best = z
    return best

# ============================================================
# هوية قمصان الفرق
# ============================================================
TEAM_COLORS = {
    "Arsenal": ("#EF0107", "#FFFFFF"),
    "Liverpool": ("#C8102E", "#FFFFFF"),
    "Chelsea": ("#034694", "#FFFFFF"),
    "Manchester City": ("#6CABDD", "#FFFFFF"),
    "Manchester United": ("#DA291C", "#FFFFFF"),
    "Tottenham Hotspur": ("#132257", "#FFFFFF"),
    "Newcastle United": ("#241F20", "#FFFFFF"),
    "Aston Villa": ("#670E36", "#FFFFFF"),
    "Brighton": ("#0057B8", "#FFFFFF"),
    "West Ham United": ("#7A263A", "#FFFFFF"),
    "Everton": ("#003399", "#FFFFFF"),
    "Crystal Palace": ("#1B458F", "#FFFFFF"),
    "Fulham": ("#FFFFFF", "#111111"),
    "Brentford": ("#E30613", "#FFFFFF"),
    "Wolverhampton Wanderers": ("#FDB913", "#111111"),
    "Bournemouth": ("#DA2915", "#FFFFFF"),
    "Nottingham Forest": ("#DD0000", "#FFFFFF"),
    "Leeds United": ("#FFCD00", "#111111"),
    "Burnley": ("#6C1D45", "#FFFFFF"),
    "Sunderland": ("#EB172B", "#FFFFFF"),
}

def team_colors(team_name):
    return TEAM_COLORS.get(str(team_name), ("#4b5563", "#FFFFFF"))

def shirt_svg(team_name, size=42):
    primary, secondary = team_colors(team_name)
    return f'''<div class="shirt-wrap">
    <svg width="{size}" height="{size}" viewBox="0 0 64 64" aria-label="قميص {html.escape(str(team_name))}">
      <path d="M18 10 L27 5 Q32 9 37 5 L46 10 L59 18 L51 31 L45 27 L45 57 L19 57 L19 27 L13 31 L5 18 Z" fill="{primary}" stroke="#111" stroke-width="1.5"/>
      <path d="M27 5 Q32 12 37 5" fill="none" stroke="{secondary}" stroke-width="3"/>
      <path d="M19 39 H45" stroke="{secondary}" stroke-width="2" opacity=".8"/>
    </svg></div>'''

# ============================================================
# الأخبار
# ============================================================
@st.cache_data(ttl=900)
def news_for_player(name, team):
    query = urllib.parse.quote_plus(f'"{name}" "{team}" football injury lineup')
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        root = ET.fromstring(r.text)
        out = []
        for item in root.findall(".//item")[:10]:
            out.append({
                "العنوان": html.unescape(item.findtext("title") or ""),
                "المصدر": item.findtext("source") or "مصدر إخباري",
                "الرابط": item.findtext("link") or "",
            })
        return out
    except Exception:
        return []

def translate_common_terms(title):
    replacements = {
        "injury":"إصابة","injured":"مصاب","fit":"جاهز","available":"متاح",
        "returns":"يعود","return":"عودة","training":"تدريب","trains":"يتدرب",
        "starting":"أساسي","starts":"يبدأ","lineup":"التشكيلة",
        "ruled out":"خارج المباراة","doubtful":"مشكوك بمشاركته",
        "suspended":"موقوف","hamstring":"أوتار الركبة","knock":"ضربة",
        "illness":"مرض","rested":"أُريح","benched":"على الدكة"
    }
    text = title
    for a,b in replacements.items():
        text = text.replace(a,b).replace(a.title(),b)
    return text

# ============================================================
# الشريط الجانبي
# ============================================================
st.sidebar.header("🎛️ لوحة تحكم Laithinho")
budget = st.sidebar.slider("الميزانية (£مليون)", 80.0, 105.0, 100.0, 0.5)

if st.sidebar.button("🔄 تحديث البيانات"):
    st.cache_data.clear()
    st.rerun()

chips = st.sidebar.multiselect(
    "البطاقات المتبقية",
    ["الوايلد كارد","الفري هِت","البنش بوست","التريبل كابتن"],
    ["الوايلد كارد","الفري هِت","البنش بوست","التريبل كابتن"]
)

st.sidebar.markdown("### 📚 مصادر النموذج")
st.sidebar.write("• الموسم الحالي: بيانات FPL الحية")
st.sidebar.write("• آخر الجولات: سجل الجولات التاريخي")
st.sidebar.write("• التاريخ: 2024/25 + 2025/26")
st.sidebar.caption("البيانات التاريخية تستخدم كمرجع، وليست بديلًا عن حالة اللاعب الحالية.")

# ============================================================
# أدوات "تشكيلتي" الشخصية
# ============================================================
FORMATION_OPTIONS = {
    "3-4-3": (3, 4, 3),
    "3-5-2": (3, 5, 2),
    "4-4-2": (4, 4, 2),
    "4-3-3": (4, 3, 3),
    "4-5-1": (4, 5, 1),
    "5-4-1": (5, 4, 1),
    "5-3-2": (5, 3, 2),
    "5-2-3": (5, 2, 3),
}

def unique_player_df(position_type=None):
    x = df.drop_duplicates("id").copy()
    if position_type is not None:
        x = x[x["element_type"] == position_type]
    return x.sort_values(["team_name", "web_name"])

def player_option_label(pid):
    r = df[df["id"] == pid]
    if r.empty:
        return "اختيار لاعب"
    r = r.iloc[0]
    return f"{r.web_name} — {r.team_name} — £{r.price:.1f}م"

def team_quality_score(my_squad, my_xi):
    if my_squad.empty or my_xi.empty:
        return 0.0
    xi_avg = float(my_xi["النقاط_المتوقعة"].mean())
    available_mask = my_squad["حالة_التوفر"] != "🔴 غير متاح"
    availability = float(my_squad.loc[available_mask, "احتمال_البداية_والتواجد"].mean()) if available_mask.any() else 0.0
    recent = float(normalize(my_squad["آخر5_متوسط"]).mean())
    hist = float(normalize(my_squad["hist_ppg"]).mean())
    fixture = float(my_squad["قوة_المباريات"].clip(0,1).mean())
    score = 100 * (
        .42 * min(xi_avg / 8.0, 1) +
        .18 * availability +
        .15 * recent +
        .13 * hist +
        .12 * fixture
    )
    return float(np.clip(score, 0, 100))

def team_diagnosis(my_squad, my_xi):
    score = team_quality_score(my_squad, my_xi)
    strengths, weaknesses = [], []
    if my_xi["النقاط_المتوقعة"].mean() >= 6.0:
        strengths.append("متوسط النقاط المتوقعة للتشكيلة الأساسية جيد.")
    if my_squad["احتمال_البداية_والتواجد"].mean() >= .85:
        strengths.append("توافر اللاعبين الأساسيين جيد.")
    if my_squad["hist_ppg"].mean() >= 3.5:
        strengths.append("الفريق لديه قاعدة تاريخية جيدة.")
    if my_squad["قوة_المباريات"].mean() >= .60:
        strengths.append("المباريات القادمة مناسبة نسبيًا.")
    if my_squad["عدد_DGW"].sum() >= 2:
        strengths.append("هناك استفادة محتملة من الـDGW.")

    weak_count = int((my_squad["مخاطرة_القرار"] >= 60).sum())
    if weak_count:
        weaknesses.append(f"هناك {weak_count} لاعب/لاعبين بقرار مرتفع المخاطرة.")
    unavailable = int((my_squad["حالة_التوفر"] == "🔴 غير متاح").sum())
    long_absent = int(my_squad["غياب_طويل_أو_غير_محدد"].sum())
    doubtful = int((my_squad["حالة_التوفر"] == "🟡 مشكوك بمشاركته").sum())
    if unavailable:
        weaknesses.append(f"لديك {unavailable} لاعب غير متاح حاليًا، وهذا خطر مباشر على الـXI.")
    if long_absent:
        weaknesses.append(f"لديك {long_absent} لاعب بحالة غياب طويل/غير محدد؛ لا نعتمد عليه في التشكيلة المثالية.")
    if doubtful:
        weaknesses.append(f"لديك {doubtful} لاعب/لاعبين مشكوك بمشاركتهم؛ راجع الأخبار قبل الموعد النهائي.")
    if my_squad["احتمال_البداية_والتواجد"].mean() < .75:
        weaknesses.append("احتمال المشاركة في الفريق أقل من المطلوب.")
    if my_squad["قوة_المباريات"].mean() < .45:
        weaknesses.append("جدول المباريات القادم ليس مثاليًا.")
    if my_squad["آخر5_متوسط"].mean() < 3.5:
        weaknesses.append("الزخم الأخير للفريق ضعيف نسبيًا.")
    if len(my_xi) == 11 and my_xi["النقاط_المتوقعة"].sum() < 55:
        weaknesses.append("سقف النقاط المتوقع للـXI يحتاج تحسينًا.")

    if not strengths:
        strengths.append("لا توجد نقطة قوة واضحة جدًا؛ الفريق متوازن.")
    if not weaknesses:
        weaknesses.append("لا توجد نقطة ضعف كبيرة حسب البيانات الحالية.")
    return score, strengths, weaknesses

def latest_team_news(my_squad, max_players=8):
    items = []
    for _, p in my_squad.sort_values("مخاطرة_القرار", ascending=False).head(max_players).iterrows():
        news = news_for_player(p.web_name, p.team_name)
        for n in news[:2]:
            items.append({
                "player": p.web_name,
                "title": translate_common_terms(n["العنوان"]),
                "source": n["المصدر"],
                "link": n["الرابط"],
            })
    return items

# ============================================================
# تبويبات الشاشة
# ============================================================
tabs = st.tabs([
    "🏟️ فريقي",
    "👤 تشكيلتي",
    "🔎 تحليل اللاعبين",
    "⚖️ مقارنة",
    "🃏 البطاقات",
    "📰 الأخبار",
    "📚 السجل التاريخي",
    "🤖 توقع ML",
    "💬 Laithinho"
])

squad = optimize_squad(budget)
xi = best_xi(squad)

# ------------------------------------------------------------
# تبويب الفريق
# ------------------------------------------------------------
with tabs[0]:
    st.subheader(f"🏟️ تشكيلة Laithinho المقترحة — الجولة {current_gw}")

    st.markdown("### 🗓️ تحليل المباريات القادمة")
    if not match_predictions.empty:
        mp = match_predictions[match_predictions["gw"] == int(current_gw)].copy()
        if mp.empty:
            mp = match_predictions.head(10).copy()
        mp["المباراة"] = mp["home"] + " 🆚 " + mp["away"]
        mp["التوقع"] = mp["xg_home"].round(2).astype(str) + " - " + mp["xg_away"].round(2).astype(str)
        mp["فوز صاحب الأرض"] = (mp["home_win"]*100).round(0).astype(int).astype(str) + "%"
        mp["كلين شيت صاحب الأرض"] = (mp["cs_home"]*100).round(0).astype(int).astype(str) + "%"
        mp["كلين شيت الضيف"] = (mp["cs_away"]*100).round(0).astype(int).astype(str) + "%"
        st.dataframe(mp[["المباراة","التوقع","فوز صاحب الأرض","كلين شيت صاحب الأرض","كلين شيت الضيف"]], use_container_width=True, hide_index=True)
    else:
        st.info("لا توجد مباريات قادمة متاحة حاليًا من مصدر FPL.")

    if xi.empty:
        st.error("لم أستطع بناء تشكيلة بهذه الميزانية.")
    else:
        captain = xi.nlargest(1, "النقاط_المتوقعة").iloc[0]
        vice = xi.nlargest(2, "النقاط_المتوقعة").iloc[1]

        st.markdown('<div class="pitch">', unsafe_allow_html=True)
        for title, typ in [("🧤 حراسة",1),("🛡️ دفاع",2),("🎯 وسط",3),("⚡ هجوم",4)]:
            group = xi[xi.element_type == typ]
            st.markdown(f"### {title}")
            cols = st.columns(max(1,len(group)))
            for j, (_, p) in enumerate(group.iterrows()):
                tag = '<span class="capt">C</span>' if p.id == captain.id else (
                    '<span class="vc">نائب</span>' if p.id == vice.id else ""
                )
                with cols[j]:
                    st.markdown(
                        f"""<div class="player">
                        {shirt_svg(p.team_name)}
                        <b>{html.escape(str(p.web_name))}</b> {tag}<br>
                        <span>{html.escape(str(p.team_name))}</span><br>
                        <span>£{p.price:.1f}م</span><br>
                        <span class="good">{p.النقاط_المتوقعة:.2f} متوقعة</span>
                        </div>""",
                        unsafe_allow_html=True
                    )
        st.markdown("</div>", unsafe_allow_html=True)

        bench = squad[~squad.index.isin(xi.index)].sort_values("النقاط_المتوقعة", ascending=False)
        st.markdown("### 🪑 الدكة")
        cols = st.columns(4)
        for j, (_, p) in enumerate(bench.iterrows()):
            with cols[j]:
                st.markdown(
                    f"""<div class="player">{shirt_svg(p.team_name)}
                    <b>{p.web_name}</b><br><span>{p.team_name}</span><br>
                    <span class="good">{p.النقاط_المتوقعة:.2f} متوقعة</span></div>""",
                    unsafe_allow_html=True
                )

        st.markdown("### 🧠 لماذا هذه التشكيلة؟")
        st.write(
            "الاختيار يجمع بين توقع المباراة القادمة، xG الفريق، Clean Sheet، Home/Away، "
            "الإنتاج الحالي، آخر 5 و10 جولات، السجل التاريخي، الدقائق والبدايات، احتمالية المشاركة وDGW."
        )

# ------------------------------------------------------------
# تبويب تشكيلتي الشخصية
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("👤 تشكيلتي — ابنِ فريقك مثل لعبة FPL وحلّله")
    st.write(
        "اختَر تشكيلتك من داخل الأداة. اختر الـ11 الأساسيين حسب المراكز ثم 4 لاعبين للدكة. "
        "بعدها Laithinho يفحص القوة، نقاط الضعف، الكابتن، الدكة، التاريخ، آخر الجولات، "
        "Home/Away، المباريات القادمة واحتمال المشاركة."
    )

    screenshot = st.file_uploader(
        "📸 اختياري: ارفع Screenshot لتشكيلتك",
        type=["png", "jpg", "jpeg", "webp"],
        key="my_team_screenshot"
    )
    if screenshot is not None:
        try:
            image = Image.open(BytesIO(screenshot.getvalue()))
            st.image(image, caption="Screenshot تشكيلتك", use_container_width=True)
            st.info(
                "الصورة تُعرض فقط في هذه النسخة. بناء التشكيلة يتم من الاختيارات أدناه "
                "حتى تكون أسماء اللاعبين دقيقة."
            )
        except Exception:
            st.error("لم أستطع قراءة الصورة. جرّب PNG أو JPG.")

    formation = st.selectbox(
        "🎮 اختر التشكيلة",
        list(FORMATION_OPTIONS.keys()),
        index=0,
        key="my_formation"
    )
    def_n, mid_n, fwd_n = FORMATION_OPTIONS[formation]

    gk_options = unique_player_df(1)["id"].astype(int).tolist()
    def_options = unique_player_df(2)["id"].astype(int).tolist()
    mid_options = unique_player_df(3)["id"].astype(int).tolist()
    fwd_options = unique_player_df(4)["id"].astype(int).tolist()

    def choose_slots(title, options, count, key_prefix):
        st.markdown(f"### {title}")
        chosen = []
        for i in range(count):
            opts = [None] + [pid for pid in options if pid not in chosen]
            pid = st.selectbox(
                f"{title} {i+1}",
                opts,
                format_func=lambda x: "— اختر لاعبًا —" if x is None else player_option_label(x),
                key=f"{key_prefix}_{i}"
            )
            if pid is not None:
                chosen.append(pid)
        return chosen

    gk_ids = choose_slots("🧤 الحراسة", gk_options, 1, "my_gk")
    def_ids = choose_slots("🛡️ الدفاع", def_options, def_n, "my_def")
    mid_ids = choose_slots("🎯 الوسط", mid_options, mid_n, "my_mid")
    fwd_ids = choose_slots("⚡ الهجوم", fwd_options, fwd_n, "my_fwd")

    st.markdown("### 🪑 الدكة")
    bench_gk_options = [x for x in gk_options if x not in gk_ids]
    bench_outfield = [x for x in (def_options + mid_options + fwd_options)
                      if x not in (def_ids + mid_ids + fwd_ids)]

    bench1 = st.selectbox(
        "الدكة 1 — حارس",
        [None] + bench_gk_options,
        format_func=lambda x: "— اختر الحارس الاحتياطي —" if x is None else player_option_label(x),
        key="my_bench_gk"
    )
    used_bench = [bench1]
    bench2_opts = [x for x in bench_outfield if x not in used_bench]
    bench2 = st.selectbox(
        "الدكة 2 — لاعب",
        [None] + bench2_opts,
        format_func=lambda x: "— اختر لاعبًا —" if x is None else player_option_label(x),
        key="my_bench_2"
    )
    used_bench.append(bench2)
    bench3_opts = [x for x in bench_outfield if x not in used_bench]
    bench3 = st.selectbox(
        "الدكة 3 — لاعب",
        [None] + bench3_opts,
        format_func=lambda x: "— اختر لاعبًا —" if x is None else player_option_label(x),
        key="my_bench_3"
    )
    used_bench.append(bench3)
    bench4_opts = [x for x in bench_outfield if x not in used_bench]
    bench4 = st.selectbox(
        "الدكة 4 — لاعب",
        [None] + bench4_opts,
        format_func=lambda x: "— اختر لاعبًا —" if x is None else player_option_label(x),
        key="my_bench_4"
    )

    all_ids = [x for x in (
        gk_ids + def_ids + mid_ids + fwd_ids + [bench1, bench2, bench3, bench4]
    ) if x is not None]

    if len(set(all_ids)) != 15:
        st.warning(f"أكمل تشكيلتك إلى 15 لاعبًا. حاليًا: {len(set(all_ids))}/15")
    else:
        my_squad = df[df["id"].isin(all_ids)].drop_duplicates("id").copy()
        my_xi_ids = [x for x in (gk_ids + def_ids + mid_ids + fwd_ids) if x is not None]
        my_xi = my_squad[my_squad["id"].isin(my_xi_ids)].copy()

        xi_options = my_xi["id"].astype(int).tolist()
        cap_id = st.selectbox(
            "👑 الكابتن", xi_options,
            format_func=player_option_label,
            key="my_captain"
        )
        vc_options = [x for x in xi_options if x != cap_id]
        vc_id = st.selectbox(
            "🥈 نائب الكابتن", vc_options,
            format_func=player_option_label,
            key="my_vc"
        )

        score, strengths, weaknesses = team_diagnosis(my_squad, my_xi)
        captain_row = my_squad[my_squad.id == cap_id].iloc[0]
        vc_row = my_squad[my_squad.id == vc_id].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("قوة الفريق", f"{score:.0f}/100")
        c2.metric("XI متوقع", f"{my_xi['النقاط_المتوقعة'].sum():.1f}")
        c3.metric("متوسط التواجد", f"{my_squad['احتمال_البداية_والتواجد'].mean()*100:.0f}%")
        c4.metric("مخاطرة القرار", f"{my_squad['مخاطرة_القرار'].mean():.0f}/100")

        unavailable_now = my_squad[my_squad["حالة_التوفر"] == "🔴 غير متاح"]
        doubtful_now = my_squad[my_squad["حالة_التوفر"] == "🟡 مشكوك بمشاركته"]
        if not unavailable_now.empty:
            st.error("🚨 **تنبيه توفر:** " + "، ".join(unavailable_now["web_name"].astype(str).tolist()) + " غير متاحين حاليًا، لذلك لا أعتمد عليهم في التشكيلة المثالية.")
        if not doubtful_now.empty:
            st.warning("🟡 **مشاركون غير مضمونين:** " + "، ".join(doubtful_now["web_name"].astype(str).tolist()) + " — راجع الأخبار قبل الموعد النهائي.")

        st.markdown("### 🩺 حالة اللاعبين الصحية/التوفر")
        availability_table = my_squad[["web_name","team_name","حالة_التوفر","نسبة_التواجد_القادمة","غياب_طويل_أو_غير_محدد","news"]].copy()
        availability_table["نسبة_التواجد_القادمة"] = (availability_table["نسبة_التواجد_القادمة"]*100).round(0).astype(int)
        availability_table.columns = ["اللاعب","الفريق","الحالة","احتمال التواجد القادم %","غياب طويل/غير محدد","ملاحظة FPL"]
        st.dataframe(availability_table, use_container_width=True, hide_index=True)

        st.markdown('<div class="pitch">', unsafe_allow_html=True)
        st.markdown(f'<div class="pitch-title">🏟️ تشكيلتك — {formation}</div>', unsafe_allow_html=True)
        for title, typ in [("🧤 حراسة",1),("🛡️ دفاع",2),("🎯 وسط",3),("⚡ هجوم",4)]:
            group = my_xi[my_xi.element_type == typ]
            st.markdown(f'<div class="row-label">{title}</div>', unsafe_allow_html=True)
            cols = st.columns(max(1, len(group)))
            for j, (_, p) in enumerate(group.iterrows()):
                tag = '<span class="capt">C</span>' if int(p.id) == int(cap_id) else (
                    '<span class="vc">نائب</span>' if int(p.id) == int(vc_id) else ''
                )
                with cols[j]:
                    st.markdown(
                        f'''<div class="player">{shirt_svg(p.team_name)}
                        <b>{html.escape(str(p.web_name))}</b> {tag}<br>
                        <span>{html.escape(str(p.team_name))}</span><br>
                        <span>£{p.price:.1f}م</span><br>
                        <span class="good">{p.النقاط_المتوقعة:.2f} متوقعة</span>
                        </div>''',
                        unsafe_allow_html=True
                    )
        st.markdown('</div>', unsafe_allow_html=True)

        bench_ids = [x for x in [bench1, bench2, bench3, bench4] if x is not None]
        my_bench = my_squad[my_squad["id"].isin(bench_ids)].copy()
        st.markdown("### 🪑 دكة فريقك")
        bcols = st.columns(4)
        for j, (_, p) in enumerate(my_bench.iterrows()):
            with bcols[j]:
                st.markdown(
                    f'''<div class="player">{shirt_svg(p.team_name)}
                    <b>{html.escape(str(p.web_name))}</b><br>
                    <span>{html.escape(str(p.team_name))}</span><br>
                    <span class="good">{p.النقاط_المتوقعة:.2f} متوقعة</span><br>
                    <span class="small">مخاطرة القرار: {p.مخاطرة_القرار:.0f}/100</span>
                    </div>''',
                    unsafe_allow_html=True
                )

        st.markdown("### 🧠 تشخيص Laithinho")
        st.markdown("**💪 نقاط القوة**")
        for s in strengths:
            st.markdown(f"- {s}")
        st.markdown("**⚠️ نقاط الضعف**")
        for w in weaknesses:
            st.markdown(f"- {w}")

        st.markdown("### 👑 قرار الكابتن")
        st.success(
            f"**{captain_row.web_name}** هو الاختيار الحالي للنموذج: "
            f"{captain_row.النقاط_المتوقعة:.2f} نقطة متوقعة، "
            f"احتمال التواجد {captain_row.احتمال_البداية_والتواجد*100:.0f}%."
        )
        st.info(
            f"النائب: **{vc_row.web_name}** — {vc_row.النقاط_المتوقعة:.2f} متوقعة. "
            "لا يعتمد القرار على آخر جولة فقط."
        )

        st.markdown("### 🔧 من أضعف الأصول عندك؟")
        weak = my_squad.sort_values(
            ["النقاط_المتوقعة", "مخاطرة_القرار"],
            ascending=[True, False]
        ).head(4)
        st.dataframe(
            weak[[
                "web_name","team_name","price","النقاط_المتوقعة","آخر5_متوسط",
                "hist_ppg","احتمال_البداية_والتواجد","قوة_المباريات","مخاطرة_القرار"
            ]],
            column_config={
                "web_name":"اللاعب","team_name":"الفريق","price":"السعر",
                "النقاط_المتوقعة":"xP الجولة","آخر5_متوسط":"متوسط آخر 5",
                "hist_ppg":"متوسط تاريخي","احتمال_البداية_والتواجد":"احتمال التواجد",
                "قوة_المباريات":"قوة المباريات","مخاطرة_القرار":"مخاطرة القرار"
            },
            use_container_width=True,
            hide_index=True
        )

        if st.checkbox(
            "📰 افحص الأخبار المؤثرة على تشكيلتي الآن",
            key="scan_my_team_news"
        ):
            with st.spinner("جاري فحص الأخبار للاعبين الأكثر حساسية في فريقك..."):
                news_items = latest_team_news(my_squad)
            if news_items:
                for n in news_items:
                    st.markdown(
                        f'''<div class="card"><b>{html.escape(n["player"])} — {html.escape(n["title"])}</b><br>
                        <span class="small">المصدر: {html.escape(n["source"])}</span><br>
                        <a href="{html.escape(n["link"])}" target="_blank">فتح الخبر الأصلي</a></div>''',
                        unsafe_allow_html=True
                    )
            else:
                st.info("لم تظهر أخبار مؤثرة من نتائج البحث الحالية.")

        st.caption(
            "التقييم يجمع البيانات الحالية + آخر 5/10 جولات + 2024/25 و2025/26 + Home/Away + "
            "الدقائق والبدايات + احتمالية المشاركة + قوة المباريات + DGW. "
            "الأخبار تُفحص عند طلبك."
        )

# ------------------------------------------------------------
# تبويب تحليل اللاعبين
# ------------------------------------------------------------
with tabs[2]:
    st.subheader("🔎 تحليل لاعب بالتفصيل")
    selected = st.selectbox("اختر اللاعب", sorted(df.web_name.dropna().unique()))
    p = df[df.web_name == selected].iloc[0]

    cols = st.columns(5)
    cols[0].metric("النقاط المتوقعة", f"{p.النقاط_المتوقعة:.2f}")
    cols[1].metric("احتمال التواجد", f"{p.احتمال_البداية_والتواجد*100:.0f}%")
    cols[2].metric("آخر 5", f"{p.آخر5_متوسط:.2f}")
    cols[3].metric("تاريخي", f"{p.hist_ppg:.2f}")
    cols[4].metric("مخاطرة القرار", f"{p.مخاطرة_القرار:.0f}/100")

    st.markdown("### 🆚 المباراة القادمة — قلب التوقع")
    st.info(
        f"**{p['المباراة_القادمة']}** — {p['المباراة_القادمة_المكان']}\n\n"
        f"أهداف الفريق المتوقعة: **{p['أهداف_الفريق_المتوقعة']:.2f}** | "
        f"احتمال الفوز: **{p['احتمال_الفوز']*100:.0f}%** | "
        f"احتمال Clean Sheet: **{p['احتمال_كلين_شيت']*100:.0f}%** | "
        f"احتمال تسجيل هدفين+: **{p['احتمال_الفريق_يسجل_هدفين']*100:.0f}%**"
    )
    st.metric("التوقع بعد فهم المباراة", f"{p['النقاط_المتوقعة']:.2f}")
    st.caption(f"توقع المباراة وحده: {p['توقع_المباراة']:.2f} | سقف تقديري: {p['سقف_النقاط_التقديري']:.1f}")

    if p["حالة_التوفر"] == "🔴 غير متاح" or bool(p["غياب_طويل_أو_غير_محدد"]):
        st.error(f"🚨 {p.web_name}: {p.حالة_التوفر}. لا يدخل التشكيلة المثالية حاليًا.")
    elif p["حالة_التوفر"] == "🟡 مشكوك بمشاركته":
        st.warning(f"🟡 {p.web_name}: المشاركة غير مضمونة. الاحتمال الحالي {p.نسبة_التواجد_القادمة*100:.0f}%.")
    else:
        st.success(f"🟢 {p.web_name}: متاح مبدئيًا للمباراة القادمة.")

    st.markdown(f"""
    <div class="card">
    <h3>{html.escape(str(p.web_name))} — {html.escape(str(p.team_name))}</h3>
    السعر: £{p.price:.1f}م<br>
    الفورمة الحالية: {p.form:.1f}<br>
    النقاط لكل مباراة حاليًا: {p.points_per_game:.1f}<br>
    xGI الحالي: {p.expected_goal_involvements:.2f}<br>
    الدقائق: {int(p.minutes)} — البدايات: {int(p.starts)}<br>
    آخر 5 جولات: {p.آخر5_نقاط:.0f} نقطة / متوسط {p.آخر5_متوسط:.2f}<br>
    آخر 5 xGI: {p.آخر5_xGI:.2f}<br>
    السجل التاريخي: {p.hist_points:.0f} نقطة في {int(p.hist_gws)} ظهور/جولة مسجلة،
    متوسط {p.hist_ppg:.2f}<br>
    <b>{risk_label(p.مخاطرة_القرار)}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏠 خارج الأرض أم داخلها؟")
    fs = upcoming(int(p.team), 6)
    if fs:
        table = pd.DataFrame([{
            "الجولة": x["gw"],
            "الخصم": x["opp"],
            "الملعب": "داخل الأرض 🏠" if x["home"] else "خارج الأرض ✈️",
            "صعوبة المباراة": x["fdr"]
        } for x in fs])
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("لا توجد مباريات قادمة متاحة في البيانات الحالية.")

    st.markdown("### 📊 سجل Home / Away التاريخي")
    home_avg = p.hist_home
    away_avg = p.hist_away
    h1,h2 = st.columns(2)
    h1.metric("متوسط داخل الأرض تاريخيًا", f"{home_avg:.2f}")
    h2.metric("متوسط خارج الأرض تاريخيًا", f"{away_avg:.2f}")

    st.markdown("### 📰 آخر الأخبار")
    news = news_for_player(p.web_name, p.team_name)
    if news:
        for n in news:
            st.markdown(
                f"""<div class="card"><b>{html.escape(translate_common_terms(n["العنوان"]))}</b><br>
                <span class="small">المصدر: {html.escape(n["المصدر"])}</span><br>
                <a href="{html.escape(n["الرابط"])}" target="_blank">فتح الخبر الأصلي</a></div>""",
                unsafe_allow_html=True
            )
    else:
        st.info("لم أجد أخبارًا حديثة لهذا اللاعب.")

# ------------------------------------------------------------
# تبويب المقارنة
# ------------------------------------------------------------
with tabs[3]:
    st.subheader("⚖️ مقارنة لاعبين")
    names = sorted(df.web_name.dropna().unique())
    a = st.selectbox("اللاعب الأول", names, key="cmp_a")
    b = st.selectbox("اللاعب الثاني", names, index=min(1,len(names)-1), key="cmp_b")

    pa = df[df.web_name == a].iloc[0]
    pb = df[df.web_name == b].iloc[0]

    comparison = pd.DataFrame({
        "المؤشر": [
            "النقاط المتوقعة","الفورمة","xGI","آخر 5 متوسط",
            "آخر 10 متوسط","المتوسط التاريخي","الدقائق",
            "البدايات","احتمال التواجد","قوة المباريات",
            "مخاطرة القرار","حالة التوفر","DGW"
        ],
        a: [
            pa.النقاط_المتوقعة,pa.form,pa.expected_goal_involvements,pa.آخر5_متوسط,
            pa.آخر10_متوسط,pa.hist_ppg,pa.minutes,pa.starts,
            pa.احتمال_البداية_والتواجد*100,pa.قوة_المباريات,pa.مخاطرة_القرار,pa.حالة_التوفر,pa.عدد_DGW
        ],
        b: [
            pb.النقاط_المتوقعة,pb.form,pb.expected_goal_involvements,pb.آخر5_متوسط,
            pb.آخر10_متوسط,pb.hist_ppg,pb.minutes,pb.starts,
            pb.احتمال_البداية_والتواجد*100,pb.قوة_المباريات,pb.مخاطرة_القرار,pb.حالة_التوفر,pb.عدد_DGW
        ]
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    winner = a if pa.النقاط_المتوقعة >= pb.النقاط_المتوقعة else b
    st.success(f"الترجيح الحالي للنموذج: **{winner}** — لكن القرار النهائي يجب أن يراجع الأخبار واحتمال البداية.")

# ------------------------------------------------------------
# تبويب البطاقات
# ------------------------------------------------------------
with tabs[4]:
    st.subheader("🃏 مستشار البطاقات")

    bench = squad[~squad.index.isin(xi.index)] if not squad.empty else pd.DataFrame()
    bench_xp = float(bench["النقاط_المتوقعة"].sum()) if not bench.empty else 0
    bb = min(100, 25 + bench_xp*4 + int((squad["عدد_DGW"] > 0).sum())*5) if not squad.empty else 0
    tc = min(100, 30 + captain["النقاط_المتوقعة"]*7 + int(captain["عدد_DGW"])*17) if not xi.empty else 0
    wc = min(100, 35 + max(0,15-len(squad))*3)
    fh = min(100, 40 + sum(1 for tid in team_names if fixture_strength(tid) < .45)*1.5)

    scores = {
        "الوايلد كارد": wc,
        "الفري هِت": fh,
        "البنش بوست": bb,
        "التريبل كابتن": tc
    }

    for chip, score in scores.items():
        status = "🔥 دراسة الاستخدام الآن" if score >= 75 else ("🟡 راقب" if score >= 55 else "⏳ احتفظ به")
        if chip not in chips:
            status = "⚫ غير متاح"
        st.markdown(f"**{chip}:** {score:.0f}/100 — {status}")

    st.info(
        "تقييم البطاقة ليس قرارًا نهائيًا. هو درجة مساعدة تعتمد على جودة التشكيلة، "
        "الـDGW، المباريات، الدكة واللاعبين المتاحين."
    )

# ------------------------------------------------------------
# تبويب الأخبار
# ------------------------------------------------------------
with tabs[5]:
    st.subheader("📰 مركز الأخبار")
    st.write("ابحث عن لاعب أو فريق لمراجعة الأخبار التي قد تؤثر على قرار FPL.")
    news_player = st.selectbox("اللاعب", sorted(df.web_name.dropna().unique()), key="news_player")
    np = df[df.web_name == news_player].iloc[0]
    news = news_for_player(np.web_name, np.team_name)
    if news:
        for n in news:
            st.markdown(
                f"""<div class="card"><b>{html.escape(translate_common_terms(n["العنوان"]))}</b><br>
                <span class="small">{html.escape(n["المصدر"])}</span><br>
                <a href="{html.escape(n["الرابط"])}" target="_blank">قراءة المصدر الأصلي</a></div>""",
                unsafe_allow_html=True
            )
    else:
        st.info("لا توجد نتائج حالية.")

    st.caption(
        "ملاحظة: هذه النسخة تعرض عنوان الخبر وتحوّل المصطلحات الشائعة. "
        "الترجمة العربية الكاملة للنص تحتاج خدمة ترجمة/API حقيقية."
    )

# ------------------------------------------------------------
# تبويب السجل التاريخي
# ------------------------------------------------------------
with tabs[6]:
    st.subheader("📚 السجل التاريخي")
    if history.empty:
        st.warning("لم تُحمّل البيانات التاريخية.")
    else:
        st.write("المواسم المستخدمة: **2024/25 و2025/26**.")

        hist_player = st.selectbox("اختر لاعبًا", sorted(df.web_name.dropna().unique()), key="hist_player")
        hp = df[df.web_name == hist_player].iloc[0]

        hcols = st.columns(4)
        hcols[0].metric("إجمالي النقاط التاريخية", f"{hp.hist_points:.0f}")
        hcols[1].metric("متوسط النقاط", f"{hp.hist_ppg:.2f}")
        hcols[2].metric("الدقائق", f"{hp.hist_minutes:.0f}")
        hcols[3].metric("ظهور/جولات مسجلة", f"{hp.hist_gws:.0f}")

        st.markdown("### آخر الجولات المسجلة")
        if "name" in history.columns and "GW" in history.columns:
            hh = history[history["name"].astype(str).str.lower() == str(hist_player).lower()].copy()
            if not hh.empty:
                cols_show = [c for c in ["الموسم","GW","total_points","minutes","goals_scored","assists","bonus","was_home"] if c in hh.columns]
                st.dataframe(hh.sort_values(["الموسم","GW"]).tail(15)[cols_show],
                             use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# تبويب Machine Learning
# ------------------------------------------------------------
with tabs[7]:
    st.subheader("🤖 محرك التوقعات ML — V6.2")
    if ml_info.get("trained"):
        a,b,c=st.columns(3)
        a.metric("حالات التدريب", f"{ml_info['rows']:,}")
        b.metric("خطأ الاختبار MAE", f"{ml_info['mae']:.2f}" if ml_info.get("mae") is not None else "—")
        c.metric("المواسم", " + ".join(ml_info.get("seasons", [])))
        st.success("محرك ML فعّال: يتعلم من بيانات تاريخية قبل الجولة، ثم يدمج توقعه مع محرك Laithinho.")
        show=df[["web_name","team_name","توقع_ML","النقاط_المتوقعة","مخاطرة_القرار"]].copy().sort_values("توقع_ML",ascending=False).head(25)
        show.columns=["اللاعب","الفريق","توقع ML","توقع Laithinho","مخاطرة القرار"]
        st.dataframe(show,use_container_width=True)
        st.info("في بداية الموسم لا نعتمد على Last 5/10 كأنها موجودة؛ GW1 تدخل كبيانات حديثة، والتاريخ السابق يعوّض قلة العينة. طبقة التوفر تستبعد المصاب/الموقوف/غير المتاح من التشكيلة المثالية.")
    else:
        st.warning("لم تتوفر بيانات تاريخية كافية لتدريب النموذج. سيستمر المحرك التقليدي بالعمل.")

# ------------------------------------------------------------
# تبويب المحادثة
# ------------------------------------------------------------
with tabs[8]:
    st.subheader("💬 تحدث مع Laithinho")

    if "chat" not in st.session_state:
        st.session_state.chat = [(
            "ai",
            "أهلًا يا ليث 👔 اسألني عن لاعب، كابتن، تبديلة أو Chip. "
            "سأحاول ربط القرار بالبيانات الحالية + آخر الجولات + التاريخ."
        )]

    for role, msg in st.session_state.chat:
        cls = "ai" if role == "ai" else "card"
        st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

    q = st.chat_input("مثال: هل أشتري اللاعب X بدل اللاعب Y؟")
    if q:
        st.session_state.chat.append(("user", q))
        low = q.lower()

        if "كابتن" in low or "captain" in low:
            top = xi.nlargest(3, "النقاط_المتوقعة")
            ans = "👑 **مرشحو الكابتن:**\n\n" + "\n".join(
                f"- {x.web_name}: {x.النقاط_المتوقعة:.2f} متوقعة، احتمال التواجد {x.احتمال_البداية_والتواجد*100:.0f}%، مخاطرة القرار {x.مخاطرة_القرار:.0f}/100"
                for _, x in top.iterrows()
            )
        elif "بنش" in low or "bench" in low:
            ans = f"🚀 تقييم البنش بوست: **{bb:.0f}/100**. الدكة الحالية حوالي **{bench_xp:.1f}** نقطة متوقعة."
        elif "تريبل" in low or "triple" in low:
            ans = f"👑 تقييم التريبل كابتن: **{tc:.0f}/100**. لا تستخدمه لمجرد DGW؛ نريد لاعبًا قويًا جدًا ودقائق مضمونة."
        elif "وايلد" in low or "wildcard" in low:
            ans = f"🃏 تقييم الوايلد كارد: **{wc:.0f}/100**. استخدمه لإعادة بناء الفريق، وليس كرد فعل على مباراة واحدة."
        else:
            ans = (
                "أعطني اسم لاعب أو لاعبين. أستطيع مقارنة النقاط المتوقعة، آخر 5/10 جولات، "
                "السجل التاريخي، Home/Away، الدقائق، البدايات، احتمال التواجد، قوة المباريات، "
                "DGW ومخاطرة القرار."
            )

        st.session_state.chat.append(("ai", ans))
        st.rerun()

st.caption(
    "Laithinho V6.2: ML + تاريخ المواسم + GW الحالية + المباريات + توفر اللاعب + اعتماد المدرب + تسريع التحميل. "
    "اللاعب غير المتاح/الغائب طويلًا لا يدخل التشكيلة المثالية، لكنه يبقى ظاهرًا في تحليل فريقك."
)
