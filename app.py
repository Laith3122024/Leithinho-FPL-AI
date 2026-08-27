
# -*- coding: utf-8 -*-
"""
Laithinho FPL AI V5
نسخة عربية متكاملة:
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
- جاهزة لاحقًا لطبقة Machine Learning حقيقية

ملاحظة:
الموسم الحالي 2026/27 يبدأ ببيانات تمهيدية، لذلك النموذج يعطي أولوية للبيانات الحالية عندما تتوفر،
ويستخدم 2025/26 و2024/25 كتاريخ مرجعي.
"""

import html
import urllib.parse
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pulp
import requests
import streamlit as st

# ============================================================
# إعداد الصفحة
# ============================================================
st.set_page_config(
    page_title="Laithinho FPL AI",
    page_icon="👔",
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
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>👔 Laithinho FPL AI</h1>
<p>تحليل الفانتسي من الجولة الحالية + آخر الجولات + تاريخ المواسم السابقة.</p>
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
    frames = []
    for season, url in HIST_URLS.items():
        try:
            h = get_historical_csv(url)
            h["الموسم"] = season
            frames.append(h)
        except Exception:
            pass
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
    rows = []
    for name in current_names:
        s = historical_player_stats(name)
        s["name"] = name
        rows.append(s)
    return pd.DataFrame(rows).set_index("name")

history_summary = build_history_summary(tuple(df["web_name"].fillna("").tolist()))

for c in history_summary.columns:
    df[c] = df["web_name"].map(history_summary[c]).fillna(0)

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
            "آخر5_xGI": last5["expected_goal_involvements"].sum(),
            "آخر10_متوسط": g.tail(10)["total_points"].mean(),
        })
    return pd.DataFrame(rows).set_index("name")

recent = recent_stats()
if not recent.empty:
    for c in recent.columns:
        df[c] = df["web_name"].map(recent[c]).fillna(0)
else:
    for c in ["آخر5_نقاط","آخر5_متوسط","آخر5_دقائق","آخر5_xGI","آخر10_متوسط"]:
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

chance_next = df["chance_of_playing_next_round"].replace(0, 100).clip(0,100) / 100
chance_this = df["chance_of_playing_this_round"].replace(0, 100).clip(0,100) / 100
df["احتمال_البداية_والتواجد"] = ((chance_next + chance_this) / 2).clip(.05,1)

df["قوة_المباريات"] = df["team"].map(fixture_strength)
df["عدد_DGW"] = df["team"].map(dgw_count)

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
    .04 * df["hist_xgi_n"]
)

df["النقاط_المتوقعة"] = (
    11.0 * (
        .72 * df["الأساس"] +
        .18 * df["ep_n"] +
        .10 * df["قوة_المباريات"].clip(0,1)
    )
).clip(.2, 13)

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
        .22 * (1 - df["starts_n"]) +
        .18 * (1 - df["minutes_n"]) +
        .14 * (1 - df["hist_ppg_n"]) +
        .14 * (1 - df["قوة_المباريات"].clip(0,1))
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
# تحسين الفريق
# ============================================================
def optimize_squad(budget):
    d = df[(df["price"] > 0) & (df["status"].isin(["a","d","i","n"]))].copy()
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
# تبويبات الشاشة
# ============================================================
tabs = st.tabs([
    "🏟️ فريقي",
    "🔎 تحليل اللاعبين",
    "⚖️ مقارنة",
    "🃏 البطاقات",
    "📰 الأخبار",
    "📚 السجل التاريخي",
    "💬 Laithinho"
])

squad = optimize_squad(budget)
xi = best_xi(squad)

# ------------------------------------------------------------
# تبويب الفريق
# ------------------------------------------------------------
with tabs[0]:
    st.subheader(f"🏟️ تشكيلة Laithinho المقترحة — الجولة {current_gw}")

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
                        <div class="shirt">👕</div>
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
                    f"""<div class="player"><div class="shirt">👕</div>
                    <b>{p.web_name}</b><br><span>{p.team_name}</span><br>
                    <span class="good">{p.النقاط_المتوقعة:.2f} متوقعة</span></div>""",
                    unsafe_allow_html=True
                )

        st.markdown("### 🧠 لماذا هذه التشكيلة؟")
        st.write(
            "الاختيار يجمع بين الإنتاج الحالي، آخر 5 و10 جولات، "
            "السجل التاريخي، الدقائق والبدايات، احتمالية المشاركة، جودة المباريات وDGW."
        )

# ------------------------------------------------------------
# تبويب تحليل اللاعبين
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("🔎 تحليل لاعب بالتفصيل")
    selected = st.selectbox("اختر اللاعب", sorted(df.web_name.dropna().unique()))
    p = df[df.web_name == selected].iloc[0]

    cols = st.columns(5)
    cols[0].metric("النقاط المتوقعة", f"{p.النقاط_المتوقعة:.2f}")
    cols[1].metric("احتمال التواجد", f"{p.احتمال_البداية_والتواجد*100:.0f}%")
    cols[2].metric("آخر 5", f"{p.آخر5_متوسط:.2f}")
    cols[3].metric("تاريخي", f"{p.hist_ppg:.2f}")
    cols[4].metric("مخاطرة القرار", f"{p.مخاطرة_القرار:.0f}/100")

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
with tabs[2]:
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
            "مخاطرة القرار","DGW"
        ],
        a: [
            pa.النقاط_المتوقعة,pa.form,pa.expected_goal_involvements,pa.آخر5_متوسط,
            pa.آخر10_متوسط,pa.hist_ppg,pa.minutes,pa.starts,
            pa.احتمال_البداية_والتواجد*100,pa.قوة_المباريات,pa.مخاطرة_القرار,pa.عدد_DGW
        ],
        b: [
            pb.النقاط_المتوقعة,pb.form,pb.expected_goal_involvements,pb.آخر5_متوسط,
            pb.آخر10_متوسط,pb.hist_ppg,pb.minutes,pb.starts,
            pb.احتمال_البداية_والتواجد*100,pb.قوة_المباريات,pb.مخاطرة_القرار,pb.عدد_DGW
        ]
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    winner = a if pa.النقاط_المتوقعة >= pb.النقاط_المتوقعة else b
    st.success(f"الترجيح الحالي للنموذج: **{winner}** — لكن القرار النهائي يجب أن يراجع الأخبار واحتمال البداية.")

# ------------------------------------------------------------
# تبويب البطاقات
# ------------------------------------------------------------
with tabs[3]:
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
with tabs[4]:
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
with tabs[5]:
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
# تبويب المحادثة
# ------------------------------------------------------------
with tabs[6]:
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
    "Laithinho V5 حاليًا Data-driven. المرحلة التالية المقترحة: تدريب ML تاريخي على بيانات Gameweek "
    "مع Backtesting، ثم دمج احتمالية البداية والأخبار في النموذج."
)
