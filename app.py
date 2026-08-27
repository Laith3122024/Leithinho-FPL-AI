
import re
import html
import math
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd
import requests
import pulp
import streamlit as st


# ============================================================
# LEITHINHO FPL AI ASSISTANT — V3
# FPL data + fixtures + news intelligence + squad optimizer
# ============================================================

st.set_page_config(
    page_title="Leithinho FPL AI",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_URL = "https://fantasy.premierleague.com/api"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Leithinho-FPL-AI/3.0)"
}

NEWS_SOURCES = [
    ("Fantasy Football Scout", "https://www.fantasyfootballscout.co.uk/"),
    ("BBC Sport Football", "https://www.bbc.com/sport/football"),
    ("Sky Sports Football", "https://www.skysports.com/football"),
    ("The Guardian Football", "https://www.theguardian.com/football"),
    ("Premier League", "https://www.premierleague.com/"),
]


# ============================================================
# UI
# ============================================================

st.markdown(
    """
    <style>
    .main { background:#0e1117; }
    .hero {
        background:linear-gradient(135deg,#171729,#24243d);
        border:1px solid #00ff87;
        border-left:6px solid #00ff87;
        padding:22px;
        border-radius:16px;
        margin-bottom:20px;
    }
    .hero h1 { color:white; margin:0; }
    .hero p { color:#00ff87; margin:5px 0 0; font-weight:bold; }
    .card {
        background:#171923;
        border:1px solid #303342;
        padding:16px;
        border-radius:14px;
        margin-bottom:12px;
    }
    .good { color:#00ff87; font-weight:bold; }
    .warn { color:#ffd166; font-weight:bold; }
    .bad { color:#ff5c7a; font-weight:bold; }
    .news {
        background:#171923;
        border-left:4px solid #00ff87;
        padding:12px;
        border-radius:8px;
        margin:7px 0;
    }
    .player-card {
        background:rgba(255,255,255,.96);
        border-radius:10px;
        padding:9px;
        text-align:center;
        box-shadow:0 4px 8px rgba(0,0,0,.25);
        margin:4px;
    }
    .player-name { font-weight:bold; color:#111; }
    .player-info { color:#555; font-size:.75rem; }
    .player-xp { color:#008000; font-weight:bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>👔 Leithinho FPL AI Assistant</h1>
      <p>FPL Data • Fixtures • News Intelligence • Squad Optimizer • Chips</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API HELPERS
# ============================================================

@st.cache_data(ttl=1800)
def api_get(endpoint):
    r = requests.get(
        f"{BASE_URL}/{endpoint}",
        headers=HEADERS,
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


# ============================================================
# FPL DATA
# ============================================================

@st.cache_data(ttl=1800)
def load_fpl():
    bootstrap = api_get("bootstrap-static/")
    fixtures = api_get("fixtures/")

    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])
    events = pd.DataFrame(bootstrap["events"])

    team_names = teams.set_index("id")["name"].to_dict()
    players["team_name"] = players["team"].map(team_names)
    players["price"] = players["now_cost"] / 10.0

    numeric_cols = [
        "form", "points_per_game", "ict_index",
        "selected_by_percent", "minutes", "starts",
        "goals_scored", "assists", "clean_sheets",
        "bonus", "bps", "influence", "creativity", "threat"
    ]

    for c in numeric_cols:
        if c in players.columns:
            players[c] = pd.to_numeric(players[c], errors="coerce").fillna(0)

    current = events[events["is_current"] == True]
    nxt = events[events["is_next"] == True]

    if not current.empty:
        current_gw = int(current.iloc[0]["id"])
    elif not nxt.empty:
        current_gw = int(nxt.iloc[0]["id"])
    else:
        current_gw = 1

    return players, teams, events, pd.DataFrame(fixtures), team_names, current_gw


try:
    df, teams_df, events_df, fixtures_df, team_names, current_gw = load_fpl()
except Exception as e:
    st.error("تعذر تحميل بيانات FPL الآن. جرّب Refresh.")
    st.code(str(e))
    st.stop()


# ============================================================
# FIXTURE ENGINE
# ============================================================

def build_fixture_map(fixtures, current_gw, horizon=5):
    result = {}

    for team_id in teams_df["id"].tolist():
        result[team_id] = []

    for _, f in fixtures.iterrows():
        event = f.get("event")
        if pd.isna(event):
            continue

        event = int(event)
        if event < current_gw or event > current_gw + horizon:
            continue

        th = int(f["team_h"])
        ta = int(f["team_a"])

        result.setdefault(th, []).append({
            "gw": event,
            "opponent": team_names.get(ta, "Unknown"),
            "difficulty": float(f.get("team_h_difficulty", 3)),
            "home": True,
        })
        result.setdefault(ta, []).append({
            "gw": event,
            "opponent": team_names.get(th, "Unknown"),
            "difficulty": float(f.get("team_a_difficulty", 3)),
            "home": False,
        })

    return result


fixture_map = build_fixture_map(fixtures_df, current_gw)

df["fixture_score"] = df["team"].map(
    lambda t: (
        sum(max(0.2, (6 - x["difficulty"]) / 5)
            for x in fixture_map.get(t, []))
        / max(len(fixture_map.get(t, [])), 1)
    )
).fillna(.5)

# Count doubles in the next 5 GWs.
def count_dgw(team_id):
    by_gw = {}
    for x in fixture_map.get(team_id, []):
        by_gw[x["gw"]] = by_gw.get(x["gw"], 0) + 1
    return sum(1 for v in by_gw.values() if v >= 2)

df["dgw_count"] = df["team"].map(count_dgw)


# ============================================================
# BASE PLAYER MODEL
# ============================================================

max_minutes = max(float(df["minutes"].max()), 1)
max_starts = max(float(df["starts"].max()), 1)

df["availability"] = (
    0.65 * (df["minutes"] / max_minutes)
    + 0.35 * (df["starts"] / max_starts)
).clip(.05, .99)

df["base_score"] = (
    df["form"] * .32
    + df["points_per_game"] * .28
    + (df["ict_index"] / 20) * .12
    + (df["bps"] / 100) * .08
    + (df["bonus"] / 20) * .05
    + df["availability"] * 2.0
)

df["xp_model"] = (
    df["base_score"]
    * (0.72 + df["fixture_score"] * .48)
).clip(0, 15)

df["captain_score"] = (
    df["xp_model"]
    * df["availability"]
    * (0.82 + df["fixture_score"] * .35)
).clip(0, 20)


# ============================================================
# NEWS INTELLIGENCE
#
# We use public RSS/Google News RSS rather than fragile scraping.
# Social platforms are represented as monitored links/searches;
# direct API ingestion can be added later with platform keys.
# ============================================================

@st.cache_data(ttl=900)
def google_news(query, max_items=8):
    encoded = urllib.parse.quote_plus(query)
    url = (
        "https://news.google.com/rss/search?"
        f"q={encoded}&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        r = requests.get(
            url,
            headers=HEADERS,
            timeout=12,
        )
        r.raise_for_status()

        root = ET.fromstring(r.text)
        rows = []

        for item in root.findall(".//item")[:max_items]:
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            pub = item.findtext("pubDate") or ""
            source = item.findtext("source") or "Google News"

            rows.append({
                "title": html.unescape(title),
                "link": link,
                "published": pub,
                "source": source,
            })

        return rows

    except Exception:
        return []


def clean_name(name):
    return re.sub(r"[^a-z0-9 ]", "", str(name).lower()).strip()


def news_sentiment(title):
    t = title.lower()

    negative_words = [
        "injury", "injured", "doubt", "doubtful", "ruled out",
        "out", "suspended", "illness", "knock", "hamstring",
        "rested", "rotation", "benched", "miss"
    ]

    positive_words = [
        "fit", "returns", "return", "training", "trains",
        "available", "starts", "starting", "boost", "ready"
    ]

    neg = sum(1 for w in negative_words if w in t)
    pos = sum(1 for w in positive_words if w in t)

    if neg > pos:
        return "negative"
    if pos > neg:
        return "positive"
    return "neutral"


def get_player_news(player_name, team_name):
    queries = [
        f'"{player_name}" "{team_name}" injury OR training OR lineup OR starts',
        f'"{player_name}" FPL',
    ]

    all_rows = []
    seen = set()

    for q in queries:
        for row in google_news(q, max_items=6):
            key = row["link"] or row["title"]
            if key not in seen:
                seen.add(key)
                all_rows.append(row)

    scored = []

    for row in all_rows[:12]:
        sentiment = news_sentiment(row["title"])

        # Source reliability is intentionally conservative.
        source_text = row["source"].lower()

        if (
            "premier league" in source_text
            or "official" in source_text
        ):
            reliability = 0.98
        elif "bbc" in source_text or "sky" in source_text:
            reliability = 0.93
        elif "guardian" in source_text or "reuters" in source_text:
            reliability = 0.92
        elif "fantasy football scout" in source_text:
            reliability = 0.90
        else:
            reliability = 0.70

        scored.append({
            **row,
            "sentiment": sentiment,
            "reliability": reliability,
        })

    return scored


# ============================================================
# NEWS IMPACT
# ============================================================

def player_news_adjustment(player_name, team_name):
    rows = get_player_news(player_name, team_name)

    if not rows:
        return {
            "adjustment": 0.0,
            "confidence": 0.50,
            "risk": "Unknown",
            "news": [],
        }

    weighted = 0
    total_weight = 0

    for r in rows:
        weight = r["reliability"]

        if r["sentiment"] == "negative":
            weighted -= 0.55 * weight
        elif r["sentiment"] == "positive":
            weighted += 0.20 * weight

        total_weight += weight

    adjustment = weighted / max(total_weight, 1)

    negatives = sum(
        1 for r in rows
        if r["sentiment"] == "negative"
    )

    positives = sum(
        1 for r in rows
        if r["sentiment"] == "positive"
    )

    if negatives >= 2:
        risk = "High"
    elif negatives == 1:
        risk = "Medium"
    elif positives >= 2:
        risk = "Low"
    else:
        risk = "Unknown"

    confidence = min(
        0.98,
        0.50 + min(len(rows), 5) * 0.08
    )

    return {
        "adjustment": adjustment,
        "confidence": confidence,
        "risk": risk,
        "news": rows,
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎛️ Control Panel")

budget = st.sidebar.slider(
    "Budget (£m)",
    80.0,
    105.0,
    100.0,
    .5,
)

team_id_input = st.sidebar.text_input(
    "FPL Team ID (optional)",
    placeholder="مثال: 1234567",
)

st.sidebar.caption(
    f"Current Gameweek: GW{current_gw}"
)

if st.sidebar.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# OPTIMIZER
# ============================================================

def solve_squad(data, max_budget):
    data = data.copy()

    data = data[
        (data["price"] > 0)
        & (data["status"].isin(["a", "d", "i", "n"]))
    ]

    # Use a small player pool for speed and stability.
    data = data.sort_values("xp_model", ascending=False).head(350)

    players = data.index.tolist()

    prob = pulp.LpProblem(
        "Leithinho_FPL",
        pulp.LpMaximize,
    )

    x = pulp.LpVariable.dicts(
        "select",
        players,
        cat="Binary",
    )

    prob += pulp.lpSum(
        data.loc[i, "xp_model"] * x[i]
        for i in players
    )

    prob += pulp.lpSum(x[i] for i in players) == 15
    prob += pulp.lpSum(
        data.loc[i, "price"] * x[i]
        for i in players
    ) <= max_budget

    for position, required in [
        (1, 2),
        (2, 5),
        (3, 5),
        (4, 3),
    ]:
        prob += pulp.lpSum(
            x[i]
            for i in players
            if data.loc[i, "element_type"] == position
        ) == required

    for team_id in data["team"].unique():
        prob += pulp.lpSum(
            x[i]
            for i in players
            if data.loc[i, "team"] == team_id
        ) <= 3

    status = prob.solve(
        pulp.PULP_CBC_CMD(msg=False)
    )

    if pulp.LpStatus[status] != "Optimal":
        return pd.DataFrame()

    selected = [
        i for i in players
        if x[i].value() == 1
    ]

    return data.loc[selected].copy()


def choose_xi(squad):
    gk = squad[squad.element_type == 1].sort_values(
        "xp_model", ascending=False
    )
    de = squad[squad.element_type == 2].sort_values(
        "xp_model", ascending=False
    )
    mi = squad[squad.element_type == 3].sort_values(
        "xp_model", ascending=False
    )
    fw = squad[squad.element_type == 4].sort_values(
        "xp_model", ascending=False
    )

    formations = [
        (3, 4, 3),
        (3, 5, 2),
        (4, 4, 2),
        (4, 3, 3),
        (4, 5, 1),
        (5, 4, 1),
        (5, 3, 2),
    ]

    best = pd.DataFrame()
    best_score = -999

    for d, m, f in formations:
        if len(de) < d or len(mi) < m or len(fw) < f:
            continue

        xi = pd.concat([
            gk.head(1),
            de.head(d),
            mi.head(m),
            fw.head(f),
        ])

        score = xi["xp_model"].sum()

        if score > best_score:
            best_score = score
            best = xi

    return best


squad = solve_squad(df, budget)

if squad.empty:
    st.error("لم أجد تشكيلة قانونية ضمن الميزانية.")
    st.stop()

starting_xi = choose_xi(squad)

bench = squad[
    ~squad.index.isin(starting_xi.index)
].sort_values(
    "xp_model",
    ascending=False
)

captain = starting_xi.sort_values(
    "captain_score",
    ascending=False
).iloc[0]

vice = starting_xi.sort_values(
    "captain_score",
    ascending=False
).iloc[1]


# ============================================================
# DASHBOARD
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric("Squad Cost", f"£{squad.price.sum():.1f}m")
c2.metric("Starting XI xP", f"{starting_xi.xp_model.sum():.1f}")
c3.metric("Captain", captain.web_name)
c4.metric("Captain Score", f"{captain.captain_score:.1f}")


# ============================================================
# PITCH
# ============================================================

st.divider()
st.subheader("⚽ Best XI")

st.markdown(
    '<div style="background:linear-gradient(135deg,#0d8f55,#20c96b);'
    'padding:18px;border-radius:18px;">',
    unsafe_allow_html=True,
)


def player_card(row):
    tags = ""
    if row["id"] == captain["id"]:
        tags += " 🟥 C"
    if row["id"] == vice["id"]:
        tags += " 🟦 VC"

    return f"""
    <div class="player-card">
      <div class="player-name">{row['web_name']}{tags}</div>
      <div class="player-info">
        {row['team_name']} • £{row['price']:.1f}m
      </div>
      <div class="player-xp">
        xP {row['xp_model']:.2f}
      </div>
    </div>
    """


for label, pos in [
    ("🧤 Goalkeeper", 1),
    ("🛡️ Defence", 2),
    ("🎯 Midfield", 3),
    ("⚡ Attack", 4),
]:
    st.markdown(f"### {label}")
    group = starting_xi[
        starting_xi.element_type == pos
    ]

    cols = st.columns(max(len(group), 1))

    for i, (_, row) in enumerate(group.iterrows()):
        with cols[i]:
            st.markdown(
                player_card(row),
                unsafe_allow_html=True
            )

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# NEWS INTELLIGENCE — TOP PLAYERS
# ============================================================

st.divider()
st.subheader("📰 Live News Intelligence")

st.caption(
    "المحرك يجمع الأخبار العامة المتاحة عبر RSS/Google News "
    "ويعطي وزنًا أعلى للمصادر الأقوى. لا يعتمد على منشور اجتماعي واحد."
)

news_candidates = pd.concat([
    starting_xi,
    bench.head(5),
]).drop_duplicates("id")

news_rows = []

for _, p in news_candidates.iterrows():
    info = player_news_adjustment(
        p["web_name"],
        p["team_name"]
    )

    adjusted_xp = (
        p["xp_model"]
        + info["adjustment"]
    )

    news_rows.append({
        "Player": p["web_name"],
        "Team": p["team_name"],
        "Model xP": round(p["xp_model"], 2),
        "News Adj.": round(info["adjustment"], 2),
        "News xP": round(adjusted_xp, 2),
        "Risk": info["risk"],
        "Confidence": f"{info['confidence']*100:.0f}%",
    })

news_df = pd.DataFrame(news_rows)

st.dataframe(
    news_df.sort_values(
        "News xP",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# CAPTAIN INTELLIGENCE
# ============================================================

st.divider()
st.subheader("👑 Captain Intelligence")

captain_rows = []

for _, p in starting_xi.iterrows():

    info = player_news_adjustment(
        p["web_name"],
        p["team_name"]
    )

    captain_rows.append({
        "Player": p["web_name"],
        "xP": round(p["xp_model"], 2),
        "Captain Score": round(p["captain_score"], 2),
        "Start Probability": f"{p['availability']*100:.0f}%",
        "Fixture": round(p["fixture_score"], 2),
        "News Risk": info["risk"],
    })

captain_df = pd.DataFrame(captain_rows)

st.dataframe(
    captain_df.sort_values(
        "Captain Score",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True,
)

st.success(
    f"👑 Current recommendation: {captain.web_name} "
    f"with {captain.captain_score:.1f} Captain Score. "
    "راجع أخبار الفريق قبل الـdeadline."
)


# ============================================================
# DIFFERENTIALS
# ============================================================

st.divider()
st.subheader("💎 Differential Radar")

diff = df[
    (df["selected_by_percent"] < 10)
    & (df["xp_model"] >= 3)
].sort_values(
    "xp_model",
    ascending=False
)

st.dataframe(
    diff[
        [
            "web_name",
            "team_name",
            "price",
            "selected_by_percent",
            "form",
            "fixture_score",
            "xp_model",
        ]
    ].head(15),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# CHIP ADVISOR
# ============================================================

st.divider()
st.subheader("🃏 Chip Intelligence")

team_counts = squad.groupby("team").size().to_dict()

dgw_exposure = 0
for team_id, count in team_counts.items():
    if count >= 1:
        dgw_exposure += count * count_dgw(team_id)

# Direct DGW discovery from the fixture list.
dgw_gws = []
for gw in range(current_gw, current_gw + 8):
    counts = {}
    matches = fixtures_df[
        fixtures_df["event"] == gw
    ]
    for _, f in matches.iterrows():
        if pd.isna(f.get("event")):
            continue
        counts[int(f["team_h"])] = counts.get(int(f["team_h"]), 0) + 1
        counts[int(f["team_a"])] = counts.get(int(f["team_a"]), 0) + 1
    if any(v >= 2 for v in counts.values()):
        dgw_gws.append(gw)

bench_xp = bench["xp_model"].sum()
bench_score = min(100, 35 + bench_xp * 5 + dgw_exposure * 4)
tc_score = min(100, 40 + len(dgw_gws) * 8 + captain.captain_score * 2)
bb_score = min(100, 30 + bench_score * .6 + dgw_exposure * 5)
wc_score = min(
    100,
    45
    + max(0, 15 - len(starting_xi)) * 1
    + len(dgw_gws) * 4
)

chip_data = [
    ("Wildcard", wc_score, "تغييرات كبيرة / إعادة بناء الفريق"),
    ("Bench Boost", bb_score, "دكة قوية + فرصة Double Gameweek"),
    ("Free Hit", min(100, 45 + len(dgw_gws) * 6),
     "مفيد عند Blank/DGW استثنائي"),
    ("Triple Captain", tc_score,
     "كابتن قوي جدًا + فرصة Double Gameweek"),
]

cols = st.columns(4)

for col, (name, score, reason) in zip(cols, chip_data):
    if score >= 75:
        status = "🔥 STRONG"
    elif score >= 55:
        status = "🟡 WATCH"
    else:
        status = "⏳ HOLD"

    with col:
        st.markdown(
            f"""
            <div class="card">
              <h3>{name}</h3>
              <h2>{score:.0f}/100</h2>
              <p>{status}</p>
              <small>{reason}</small>
              <br><br>
              <small>Potential DGWs: {dgw_gws or 'None detected'}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PLAYER NEWS DETAIL
# ============================================================

st.divider()
st.subheader("🔎 Deep News Check")

selected_name = st.selectbox(
    "اختر لاعبًا لعرض الأخبار",
    sorted(
        df["web_name"].dropna().unique().tolist()
    ),
)

selected = df[
    df["web_name"] == selected_name
].sort_values(
    "xp_model",
    ascending=False
).iloc[0]

detail = player_news_adjustment(
    selected["web_name"],
    selected["team_name"]
)

st.markdown(
    f"""
    **{selected['web_name']} — {selected['team_name']}**

    Model xP: **{selected['xp_model']:.2f}**

    Start probability: **{selected['availability']*100:.0f}%**

    Fixture score: **{selected['fixture_score']:.2f}**

    News risk: **{detail['risk']}**

    News confidence: **{detail['confidence']*100:.0f}%**
    """
)

if detail["news"]:
    for item in detail["news"][:8]:
        st.markdown(
            f"""
            <div class="news">
              <b>{item['title']}</b><br>
              <small>
                {item['source']} •
                Reliability {item['reliability']*100:.0f}% •
                {item['sentiment']}
              </small><br>
              <a href="{item['link']}" target="_blank">Open source</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("لم أجد أخبارًا حديثة كافية لهذا اللاعب.")


# ============================================================
# SOCIAL HUB
# ============================================================

st.divider()
st.subheader("📱 Social / Official Sources")

st.write(
    "هذه الروابط تفتح المصادر الرسمية والمختصة مباشرة. "
    "لا نعتمد على scraping غير رسمي لمنصات التواصل، لأن الوصول "
    "إليها قد يتغير أو يحتاج API keys."
)

social_links = [
    ("Official FPL", "https://x.com/OfficialFPL"),
    ("Premier League", "https://x.com/premierleague"),
    ("Fantasy Football Scout", "https://x.com/FFScout"),
    ("FPL YouTube Search", "https://www.youtube.com/results?search_query=Fantasy+Premier+League+team+news"),
    ("FPL Instagram Search", "https://www.instagram.com/explore/search/keyword/?q=fpl"),
]

for name, url in social_links:
    st.markdown(f"- [{name}]({url})")


# ============================================================
# OPTIONAL TEAM ID
# ============================================================

if team_id_input.strip():
    try:
        team_id = int(team_id_input.strip())

        picks_payload = api_get(
            f"entry/{team_id}/event/{current_gw}/picks/"
        )

        picks = pd.DataFrame(
            picks_payload.get("picks", [])
        )

        if not picks.empty:
            st.divider()
            st.subheader("👤 Your Team Snapshot")

            mine = picks.merge(
                df,
                left_on="element",
                right_on="id",
                how="left",
            )

            st.dataframe(
                mine[
                    [
                        "web_name",
                        "team_name",
                        "price",
                        "xp_model",
                        "captain_score",
                        "multiplier",
                    ]
                ].sort_values(
                    "xp_model",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True,
            )

            my_xp = mine["xp_model"].sum()
            optimal_xp = starting_xi["xp_model"].sum()

            st.info(
                f"Your 15-player model xP: {my_xp:.1f} | "
                f"Optimal XI model xP: {optimal_xp:.1f}"
            )

    except Exception:
        st.warning(
            "تعذر تحميل Team ID. تأكد أنه رقم صحيح."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Leithinho FPL AI • V3 • Data-driven assistant. "
    "News signals are probabilistic; always verify final team news "
    "and official FPL information before deadline."
)
