from __future__ import annotations

from datetime import datetime, timedelta
from html import escape
from math import isfinite
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots


# -----------------------------
# Data sources
# -----------------------------
TIMEZONE = "Europe/Warsaw"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_MODEL = "ecmwf_ifs"

WARSAW = {
    "name": "Warszawa",
    "latitude": 52.239,
    "longitude": 21.079,
}

INPOST_URL = "https://inpost.pl/shipx-point-data/40203/WAW582M/air_index_level"
INPOST_REFERER = "https://inpost.pl/paczkomat-warszawa-waw582m-brazylijska-paczkomaty-mazowieckie"

HOURLY_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "precipitation_probability",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "relative_humidity_2m",
    "surface_pressure",
    "weather_code",
    "cloud_cover",
]


# -----------------------------
# Helpers
# -----------------------------
def now_warsaw() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None)


def weather_info(code: Any) -> tuple[str, str]:
    if pd.isna(code):
        return "Brak danych", "🌡️"

    code = int(code)
    mapping = {
        0: ("Bezchmurnie", "☀️"),
        1: ("Przeważnie bezchmurnie", "🌤️"),
        2: ("Częściowe zachmurzenie", "⛅"),
        3: ("Pochmurno", "☁️"),
        45: ("Mgła", "🌫️"),
        48: ("Mgła osadzająca szadź", "🌫️"),
        51: ("Lekka mżawka", "🌦️"),
        53: ("Mżawka", "🌦️"),
        55: ("Silna mżawka", "🌧️"),
        61: ("Lekki deszcz", "🌦️"),
        63: ("Deszcz", "🌧️"),
        65: ("Silny deszcz", "🌧️"),
        71: ("Lekki śnieg", "🌨️"),
        73: ("Śnieg", "🌨️"),
        75: ("Silny śnieg", "❄️"),
        80: ("Przelotny deszcz", "🌦️"),
        81: ("Przelotny deszcz", "🌧️"),
        82: ("Silne opady przelotne", "🌧️"),
        95: ("Burza", "⛈️"),
        96: ("Burza z gradem", "⛈️"),
        99: ("Silna burza z gradem", "⛈️"),
    }
    return mapping.get(code, (f"WMO {code}", "🌡️"))


def cardinal(deg: Any) -> str:
    if pd.isna(deg):
        return ""
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return directions[round(float(deg) / 45) % 8]


def _fmt(value: Any, digits: int = 0, fallback: str = "—") -> str:
    if value is None or pd.isna(value):
        return fallback
    return f"{float(value):.{digits}f}"


@st.cache_data(ttl=900, show_spinner=False)
def fetch_forecast(latitude: float, longitude: float) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": TIMEZONE,
        "forecast_days": 10,
        "models": FORECAST_MODEL,
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "hourly": ",".join(HOURLY_VARS),
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()
    hourly = payload.get("hourly", {})
    if not isinstance(hourly, dict) or not hourly.get("time"):
        raise RuntimeError("Open-Meteo nie zwróciło danych godzinowych.")
    missing = set(HOURLY_VARS) - hourly.keys()
    if missing:
        raise RuntimeError(f"Open-Meteo: brak wymaganych pól: {', '.join(sorted(missing))}.")
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_inpost_actual() -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "checked_at": now_warsaw().isoformat(timespec="seconds"),
        "method": "POST",
        "endpoint": INPOST_URL,
    }
    headers = {
        "X-Requested-With": "XMLHttpRequest",
        "Referer": INPOST_REFERER,
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }
    try:
        r = requests.post(INPOST_URL, headers=headers, timeout=12)
        diagnostics.update({
            "http_status": r.status_code,
            "response_url": r.url,
            "content_type": r.headers.get("Content-Type", ""),
            "elapsed_seconds": round(r.elapsed.total_seconds(), 3),
            "redirects": [response.status_code for response in r.history],
            "response_preview": r.text[:1500],
        })
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, dict):
            raise ValueError("InPost zwrócił nieprawidłowy format danych.")
        diagnostics["json_keys"] = list(payload.keys())
        diagnostics["air_sensors_type"] = type(payload.get("air_sensors")).__name__
        sensors: dict[str, float] = {}
        for item in payload.get("air_sensors") or []:
            if not isinstance(item, str):
                continue
            parts = item.split(":")
            if len(parts) >= 2 and parts[1]:
                try:
                    value = float(parts[1])
                    if isfinite(value):
                        sensors[parts[0].strip().upper()] = value
                except ValueError:
                    pass
        if not sensors:
            raise ValueError("InPost nie zwrócił odczytów czujników.")
        diagnostics["parsed_sensors"] = sensors
        return {
            "temperature": sensors.get("TEMPERATURE"),
            "humidity": sensors.get("HUMIDITY"),
            "pressure": sensors.get("PRESSURE"),
            "pm1": sensors.get("PM1"),
            "pm25": sensors.get("PM25"),
            "pm10": sensors.get("PM10"),
            "air_quality": payload.get("air_index_level", "—"),
            "time": now_warsaw(),
            "source": "WAW582M · InPost",
        }, None, diagnostics
    except (requests.RequestException, ValueError, TypeError) as exc:
        diagnostics["exception_type"] = type(exc).__name__
        diagnostics["exception"] = str(exc)
        return None, f"{type(exc).__name__}: {exc}", diagnostics


def nearest_forecast_row(df: pd.DataFrame, when: datetime) -> pd.Series:
    idx = (df["time"] - when).abs().idxmin()
    return df.loc[idx]


def next_hours(df: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    current = now_warsaw()
    start = current.replace(minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=hours)
    out = df[(df["time"] >= start) & (df["time"] <= end)].copy()
    return out.head(hours + 1)


def daily_summary(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    current_date = now_warsaw().date()
    work = df[df["date"] >= current_date].copy()
    rows: list[dict[str, Any]] = []
    for d, g in work.groupby("date"):
        mode = g["weather_code"].mode()
        code = mode.iloc[0] if not mode.empty else None
        label, icon = weather_info(code)
        rows.append(
            {
                "date": d,
                "min": float(pd.to_numeric(g["temperature_2m"], errors="coerce").min()),
                "max": float(pd.to_numeric(g["temperature_2m"], errors="coerce").max()),
                "precip": float(pd.to_numeric(g["precipitation"], errors="coerce").sum(min_count=1)),
                "pop": float(pd.to_numeric(g["precipitation_probability"], errors="coerce").max()),
                "label": label,
                "icon": icon,
            }
        )
    return pd.DataFrame(rows).head(days)


def translate_air_quality(value: str | None) -> str:
    mapping = {
        "VERY_GOOD": "BARDZO DOBRA",
        "GOOD": "DOBRA",
        "MODERATE": "UMIARKOWANA",
        "SUFFICIENT": "DOSTATECZNA",
        "BAD": "ZŁA",
        "VERY_BAD": "BARDZO ZŁA",
    }
    return mapping.get(str(value or "").upper(), str(value or "—").replace("_", " "))


# -----------------------------
# Charts
# -----------------------------
def style_plot(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=26, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DDE9F7", size=12),
        hoverlabel=dict(bgcolor="#10243D", font_color="#F8FBFF"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(showgrid=False, color="#8EA7C3", tickformat="%H:%M")
    fig.update_yaxes(gridcolor="rgba(132,160,190,.14)", zeroline=False, color="#8EA7C3")
    return fig


def temperature_chart(hourly: pd.DataFrame, actual: dict[str, Any] | None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hourly["time"],
            y=hourly["temperature_2m"],
            mode="lines+markers",
            name="Prognoza",
            line=dict(color="#53A7FF", width=3),
            marker=dict(size=4, color="#53A7FF"),
            hovertemplate="%{x|%H:%M}<br>%{y:.1f}°C<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hourly["time"],
            y=hourly["apparent_temperature"],
            mode="lines",
            name="Odczuwalna",
            line=dict(color="#7CE0C3", width=2, dash="dot"),
            hovertemplate="%{x|%H:%M}<br>Odczuwalna %{y:.1f}°C<extra></extra>",
        )
    )
    if actual and actual.get("temperature") is not None:
        fig.add_trace(
            go.Scatter(
                x=[actual["time"]],
                y=[actual["temperature"]],
                mode="markers+text",
                name="Teraz · WAW582M",
                text=[f"{actual['temperature']:.1f}°"],
                textposition="top center",
                marker=dict(size=13, color="#FF9D3D", line=dict(color="#FFD5A0", width=2)),
                textfont=dict(color="#FFD5A0", size=12),
                hovertemplate="WAW582M<br>%{y:.2f}°C<extra></extra>",
            )
        )
    fig.update_yaxes(title="°C")
    return style_plot(fig, 315)


def rain_chart(hourly: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=hourly["time"],
            y=hourly["precipitation"],
            name="Opad",
            marker_color="#3F8CFF",
            opacity=0.88,
            hovertemplate="%{x|%H:%M}<br>%{y:.1f} mm<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=hourly["time"],
            y=hourly["precipitation_probability"],
            name="POP",
            mode="lines",
            line=dict(color="#B08CFF", width=2),
            hovertemplate="%{x|%H:%M}<br>%{y:.0f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title="mm", secondary_y=False)
    fig.update_yaxes(title="%", range=[0, 100], secondary_y=True, showgrid=False)
    return style_plot(fig, 235)


def wind_chart(hourly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=hourly["time"],
            y=hourly["wind_speed_10m"],
            name="Wiatr",
            marker_color="#35C991",
            opacity=0.85,
            hovertemplate="%{x|%H:%M}<br>%{y:.0f} km/h<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hourly["time"],
            y=hourly["wind_gusts_10m"],
            name="Porywy",
            mode="lines",
            line=dict(color="#F2C14E", width=2),
            hovertemplate="%{x|%H:%M}<br>Porywy %{y:.0f} km/h<extra></extra>",
        )
    )
    fig.update_yaxes(title="km/h")
    return style_plot(fig, 235)


def air_chart(actual: dict[str, Any]) -> go.Figure:
    labels = ["PM1", "PM2.5", "PM10"]
    values = [actual.get("pm1"), actual.get("pm25"), actual.get("pm10")]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=["#70D86B", "#E7C64E", "#FF8A47"],
            text=["—" if v is None else f"{v:.1f}" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>%{y:.1f} µg/m³<extra></extra>",
        )
    )
    valid_values = [float(v) for v in values if v is not None and pd.notna(v) and isfinite(float(v))]
    upper = max(10.0, max(valid_values, default=0.0) * 1.3)
    fig.update_yaxes(title="µg/m³", range=[0, upper], autorange=False)
    return style_plot(fig, 230)


def daily_temperature_chart(daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["max"],
            mode="lines+markers",
            name="Max",
            line=dict(color="#FF9D3D", width=3),
            marker=dict(size=7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["min"],
            mode="lines+markers",
            name="Min",
            line=dict(color="#4CA6FF", width=3),
            marker=dict(size=7),
            fill="tonexty",
            fillcolor="rgba(76,166,255,.09)",
        )
    )
    fig.update_yaxes(title="°C")
    style_plot(fig, 300)
    fig.update_xaxes(tickformat="%d.%m")
    return fig


# -----------------------------
# UI building blocks
# -----------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #06111f;
            --panel: #0b1b2f;
            --panel2: #10243d;
            --line: rgba(147, 181, 215, .14);
            --muted: #91a8c1;
            --text: #f3f8ff;
            --blue: #4da2ff;
        }
        html, body, [class*="css"] { background: var(--bg); }
        .stApp { background: linear-gradient(180deg, #071321 0%, #06111f 55%, #05101c 100%); }
        .block-container {
            max-width: 760px;
            padding-top: 4.5rem;
            padding-left: .8rem;
            padding-right: .8rem;
            padding-bottom: 4.5rem;
        }
        header[data-testid="stHeader"] { background: rgba(6,17,31,.92); }
        [data-testid="stToolbar"] { visibility: hidden; height: 0; }
        .app-top {
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:.8rem;
            padding:.35rem .15rem .65rem .15rem;
        }
        .location-title {font-size:1.2rem;font-weight:850;color:var(--text);letter-spacing:-.02em;}
        .location-sub {font-size:.78rem;color:var(--muted);margin-top:.05rem;}
        .status-dot {width:.55rem;height:.55rem;border-radius:99px;background:#4bd873;display:inline-block;margin-right:.4rem;box-shadow:0 0 12px rgba(75,216,115,.6);}
        .source-pill {font-size:.72rem;color:#bdd2e8;border:1px solid var(--line);padding:.35rem .55rem;border-radius:999px;background:#0b1b2f;text-align:center;}
        .hero-card {
            border-radius:22px;
            padding:1rem 1rem .95rem 1rem;
            background: linear-gradient(145deg, #102844 0%, #0b1d33 54%, #0e2741 100%);
            border:1px solid rgba(112,164,211,.18);
            box-shadow:0 14px 34px rgba(0,0,0,.24);
            margin:.25rem 0 .8rem 0;
        }
        .hero-source {color:#b8cee5;font-size:.78rem;font-weight:700;display:flex;align-items:center;gap:.35rem;}
        .hero-main {display:flex;align-items:center;justify-content:space-between;gap:1rem;margin:.65rem 0 .85rem 0;}
        .hero-temp {font-size:3.2rem;line-height:.95;font-weight:820;color:#f7fbff;letter-spacing:-.06em;}
        .hero-temp span {font-size:1.15rem;vertical-align:top;margin-left:.1rem;color:#d5e7f8;letter-spacing:0;}
        .hero-condition {color:#d8e7f6;font-size:.95rem;margin-top:.35rem;}
        .hero-feels {color:#8ea8c3;font-size:.78rem;margin-top:.12rem;}
        .weather-emoji {font-size:3.2rem;filter:drop-shadow(0 8px 16px rgba(0,0,0,.25));}
        .aq-badge {display:inline-flex;align-items:center;gap:.35rem;border-radius:999px;background:rgba(31,151,84,.26);border:1px solid rgba(66,208,116,.2);padding:.42rem .65rem;color:#91f0b3;font-size:.74rem;font-weight:850;}
        .hero-metrics {display:grid;grid-template-columns:repeat(4,1fr);gap:.4rem;border-top:1px solid var(--line);padding-top:.75rem;}
        .hero-metric {min-width:0;}
        .hero-metric .v {color:#f4f9ff;font-weight:850;font-size:.97rem;white-space:nowrap;}
        .hero-metric .l {color:#819bb6;font-size:.65rem;margin-top:.08rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
        .section-card {background:#0a192b;border:1px solid var(--line);border-radius:18px;padding:.55rem .65rem .25rem .65rem;margin:.5rem 0 .7rem 0;box-shadow:0 10px 24px rgba(0,0,0,.16);}
        .section-title {font-size:1rem;font-weight:850;color:#f3f8ff;margin:.1rem .15rem .25rem .15rem;}
        .section-sub {font-size:.72rem;color:#829ab3;margin:0 .15rem .35rem .15rem;}
        .hour-strip {display:grid;grid-template-columns:repeat(6,1fr);gap:.35rem;margin:.3rem 0 .8rem 0;overflow-x:auto;}
        .hour-card {background:#0a192b;border:1px solid var(--line);border-radius:14px;text-align:center;padding:.55rem .2rem;min-width:72px;}
        .hour-time {font-size:.68rem;color:#8ba3bd;}
        .hour-icon {font-size:1.25rem;margin:.2rem 0;}
        .hour-temp {font-size:.92rem;font-weight:850;color:#f6fbff;}
        .hour-pop {font-size:.62rem;color:#65b6ff;margin-top:.12rem;}
        .daily-row {display:grid;grid-template-columns:1.1fr .45fr .95fr .8fr;align-items:center;gap:.3rem;padding:.7rem .2rem;border-bottom:1px solid rgba(133,163,192,.10);}
        .daily-day {font-weight:800;color:#f5f9ff;font-size:.88rem;}
        .daily-icon {font-size:1.2rem;text-align:center;}
        .daily-temp {font-weight:850;color:#dfeaf5;text-align:center;font-size:.87rem;}
        .daily-rain {color:#79bfff;text-align:right;font-size:.76rem;}
        .tiny-note {color:#7089a3;font-size:.7rem;line-height:1.35;margin:.25rem 0 .55rem 0;}
        div[data-testid="stTabs"] button {font-weight:800;color:#91a8c1;padding-left:.6rem;padding-right:.6rem;}
        div[data-testid="stTabs"] button[aria-selected="true"] {color:#f7fbff;}
        div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {background-color:#3f9dff;}
        .stPlotlyChart {background:#0a192b;border-radius:18px;border:1px solid var(--line);padding:.12rem .18rem;}
        .stAlert {border-radius:14px;}
        @media (max-width: 560px) {
            .block-container {padding-left:.55rem;padding-right:.55rem;}
            .hero-card {border-radius:18px;padding:.9rem .85rem;}
            .hero-temp {font-size:2.75rem;}
            .weather-emoji {font-size:2.75rem;}
            .hero-metrics {grid-template-columns:repeat(4,1fr);gap:.15rem;}
            .hero-metric .v {font-size:.83rem;}
            .hero-metric .l {font-size:.58rem;}
            .hour-strip {grid-template-columns:repeat(6,74px);}
            .daily-row {grid-template-columns:1.15fr .42fr .9fr .8fr;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        f"""
        <div class="app-top">
          <div>
            <div class="location-title">⌖ Warszawa</div>
            <div class="location-sub">Praga-Południe · {now_warsaw():%H:%M}</div>
          </div>
          <div class="source-pill">ECMWF IFS + WAW582M</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(actual: dict[str, Any] | None, current_fc: pd.Series) -> None:
    condition, emoji = weather_info(current_fc.get("weather_code"))
    temp = actual.get("temperature") if actual and actual.get("temperature") is not None else current_fc.get("temperature_2m")
    humidity = actual.get("humidity") if actual and actual.get("humidity") is not None else current_fc.get("relative_humidity_2m")
    pressure = actual.get("pressure") if actual and actual.get("pressure") is not None else current_fc.get("surface_pressure")
    pm25 = actual.get("pm25") if actual else None
    pm10 = actual.get("pm10") if actual else None
    aq = translate_air_quality(actual.get("air_quality") if actual else None)
    source = "WAW582M · InPost" if actual else "ECMWF IFS · Open-Meteo"
    actual_time = actual.get("time") if actual else now_warsaw()

    st.markdown(
        f"""
        <div class="hero-card">
          <div class="hero-source"><span class="status-dot"></span>{escape(source)} · {actual_time:%H:%M}</div>
          <div class="hero-main">
            <div>
              <div class="hero-temp">{_fmt(temp,1)}<span>°C</span></div>
              <div class="hero-condition">{escape(condition)}</div>
              <div class="hero-feels">Odczuwalna {_fmt(current_fc.get('apparent_temperature'),1)}°C</div>
            </div>
            <div style="text-align:right">
              <div class="weather-emoji">{emoji}</div>
              <div class="aq-badge">● {escape(aq)}</div>
            </div>
          </div>
          <div class="hero-metrics">
            <div class="hero-metric"><div class="v">💧 {_fmt(humidity,0)}%</div><div class="l">Wilgotność</div></div>
            <div class="hero-metric"><div class="v">◴ {_fmt(pressure,0)}</div><div class="l">hPa</div></div>
            <div class="hero-metric"><div class="v">◌ {_fmt(pm25,1)}</div><div class="l">PM2.5</div></div>
            <div class="hero-metric"><div class="v">◉ {_fmt(pm10,1)}</div><div class="l">PM10</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hour_strip(hourly: pd.DataFrame) -> None:
    cards = []
    for _, row in hourly.head(6).iterrows():
        _, icon = weather_info(row["weather_code"])
        cards.append(
            f"""
            <div class="hour-card">
              <div class="hour-time">{row['time']:%H:%M}</div>
              <div class="hour-icon">{icon}</div>
              <div class="hour-temp">{_fmt(row['temperature_2m'])}°</div>
              <div class="hour-pop">💧 {_fmt(row['precipitation_probability'])}%</div>
            </div>
            """
        )
    # Keep HTML on one line so Markdown cannot interpret indented cards as code.
    html = '<div class="hour-strip">' + "".join(cards) + "</div>"
    st.markdown("".join(line.strip() for line in html.splitlines()), unsafe_allow_html=True)


def render_daily_list(daily: pd.DataFrame) -> None:
    pl_days = ["Pon", "Wt", "Śr", "Czw", "Pt", "Sob", "Nd"]
    rows = []
    today = now_warsaw().date()
    for _, row in daily.iterrows():
        d = row["date"]
        label = "Dzisiaj" if d == today else pl_days[d.weekday()]
        rows.append(
            f"""
            <div class="daily-row">
              <div class="daily-day">{label}</div>
              <div class="daily-icon">{row['icon']}</div>
              <div class="daily-temp">{_fmt(row['min'],0)}° / {_fmt(row['max'],0)}°</div>
              <div class="daily-rain">💧 {_fmt(row['pop'],0)}% · {_fmt(row['precip'],1)} mm</div>
            </div>
            """
        )
    html = '<div class="section-card">' + "".join(rows) + "</div>"
    st.markdown("".join(line.strip() for line in html.splitlines()), unsafe_allow_html=True)


# -----------------------------
# App
# -----------------------------
def main() -> None:
    st.set_page_config(
        page_title="Warszawa Weather",
        page_icon="🌤️",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    render_header()

    try:
        forecast = fetch_forecast(WARSAW["latitude"], WARSAW["longitude"])
    except Exception as exc:
        st.error(f"Nie udało się pobrać prognozy Open-Meteo: {exc}")
        return

    if st.button("Odśwież odczyt InPost", key="refresh_inpost"):
        fetch_inpost_actual.clear()
    actual, actual_error, diagnostics = fetch_inpost_actual()
    if actual_error:
        st.warning(f"Brak pomiaru InPost. Wyświetlane dane pogodowe są prognozą ECMWF. Błąd: {actual_error}")
    with st.expander("Diagnostyka InPost", expanded=bool(actual_error)):
        st.caption("Odpowiedź z serwera uruchamiającego aplikację. Czas oznacza pobranie, nie czas pomiaru czujnika. Wynik jest buforowany przez 5 minut; przycisk powyżej wymusza nową próbę.")
        st.json(diagnostics)
    current_fc = nearest_forecast_row(forecast, now_warsaw())
    hourly = next_hours(forecast, 24)
    daily = daily_summary(forecast, 7)

    tab_now, tab_hourly, tab_daily = st.tabs(["Teraz", "Godzinowa", "Dzienna"])

    with tab_now:
        render_hero(actual, current_fc)
        if actual_error:
            st.caption("WAW582M chwilowo niedostępny — karta TERAZ korzysta z fallbacku Open-Meteo.")

        render_hour_strip(hourly)

        st.markdown('<div class="section-title">Temperatura</div><div class="section-sub">Prognoza godzinowa + punkt pomiarowy WAW582M</div>', unsafe_allow_html=True)
        st.plotly_chart(temperature_chart(hourly, actual), key="weather_chart_1", use_container_width=True, config={"displayModeBar": False})

        col_rain, col_wind = st.columns(2, gap="small")
        with col_rain:
            st.markdown('<div class="section-title">Opady</div>', unsafe_allow_html=True)
            st.plotly_chart(rain_chart(hourly), key="weather_chart_2", use_container_width=True, config={"displayModeBar": False})
        with col_wind:
            st.markdown('<div class="section-title">Wiatr</div>', unsafe_allow_html=True)
            st.plotly_chart(wind_chart(hourly), key="weather_chart_3", use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-title">Jakość powietrza</div><div class="section-sub">Bieżący odczyt czujnika WAW582M</div>', unsafe_allow_html=True)
        if actual:
            st.plotly_chart(air_chart(actual), key="weather_chart_4", use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Brak bieżącego odczytu WAW582M.")

    with tab_hourly:
        render_hero(actual, current_fc)
        st.markdown('<div class="section-title">Najbliższe 24 godziny</div>', unsafe_allow_html=True)
        render_hour_strip(hourly)
        st.plotly_chart(temperature_chart(hourly, actual), key="weather_chart_5", use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(rain_chart(hourly), key="weather_chart_6", use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(wind_chart(hourly), key="weather_chart_7", use_container_width=True, config={"displayModeBar": False})

        with st.expander("Dane godzinowe"):
            table = hourly[
                [
                    "time",
                    "temperature_2m",
                    "apparent_temperature",
                    "precipitation",
                    "precipitation_probability",
                    "wind_speed_10m",
                    "wind_gusts_10m",
                    "relative_humidity_2m",
                    "surface_pressure",
                    "cloud_cover",
                ]
            ].copy()
            table["time"] = table["time"].dt.strftime("%H:%M")
            table.columns = [
                "Godzina",
                "Temp °C",
                "Odczuwalna °C",
                "Opad mm",
                "POP %",
                "Wiatr km/h",
                "Porywy km/h",
                "Wilgotność %",
                "Ciśnienie hPa",
                "Chmury %",
            ]
            st.dataframe(table, hide_index=True, width="stretch")

    with tab_daily:
        st.markdown('<div class="section-title">7 dni</div><div class="section-sub">Min/max temperatury i opady</div>', unsafe_allow_html=True)
        st.plotly_chart(daily_temperature_chart(daily), key="weather_chart_8", use_container_width=True, config={"displayModeBar": False})
        render_daily_list(daily)

    st.markdown(
        '<div class="tiny-note">Pomiar: WAW582M / InPost. Prognoza: ECMWF IFS przez Open-Meteo. Odczyt czujnika może być chwilowo niedostępny; wtedy aplikacja przełącza kartę „Teraz” na dane prognozowe.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
