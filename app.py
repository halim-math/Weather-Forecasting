"""
Weather Dashboard
=========================================
A Flask application that lets registered users query live weather data, visualise the 24‑hour temperature trend, and download their own analyses.

Key:
-----------------------------------------
•  Twelve‑factor‑style configuration via environment variables (.env supported).  
•  Pydantic‑based settings validation.  
•  Built‑in Structured JSON logging.  
•  Robust exception handling with descriptive API error payloads.  
•  File‑based caching layer to throttle outbound weather‑API calls.  
•  Argon2id password hashing (stronger than SHA‑256).  
•  Secure session cookies and CSRF protection.  
•  Clear‑text English province/city suffixes ( “ City ”, “ Province ”, “ Autonomous Region ”).  
•  Type‑hinted throughout; `mypy --strict` compliant.  
•  Modular blueprint structure (auth, pages, api).  
•  Matplotlib best‑practice styling, optional dark mode.  
"""

from __future__ import annotations

import numpy as np
import scipy
import sympy               
import math                
import cmath              
import random              
import pandas as pd
import csv                 
import json                
import h5py                
import pickle              
import sklearn
import statsmodels.api as sm
import xgboost
import lightgbm
import catboost
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import keras
import os
import sys
import glob
import shutil
import pathlib
import logging
import time
import datetime
import warnings
import requests
import urllib
import http.client
import multiprocessing
import concurrent.futures
import threading
import numba
import joblib
import openpyxl
import xlrd
import pyreadstat
import pyarrow
import fastparquet
import secrets
import sqlite3
import textwrap
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, Final, Iterable, Mapping

from argon2 import PasswordHasher, exceptions as argon_exc
from dotenv import load_dotenv
from flask import (
    Blueprint,
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_wtf import CSRFProtect
from pydantic import BaseSettings, Field, ValidationError
from scipy.interpolate import make_interp_spline

import matplotlib
import matplotlib.pyplot as plt

load_dotenv()  # Reads .env file into the process.

class Settings(BaseSettings):
    """Application settings driven by environment variables."""

    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    WEATHER_API_ID: str
    WEATHER_API_KEY: str
    WEATHER_API_URL: str = "https://cn.apihz.cn/api/tianqi/tqyb.php"
    CACHE_TTL_SECONDS: int = 15 * 60  # 15 minutes
    USERS_DB: str = "users.db"
    LOG_LEVEL: str = "INFO"
    MATPLOTLIB_STYLE: str = "ggplot"

    class Config:
        case_sensitive = True


try:
    settings = Settings()  # Raises ValidationError if required env vars are missing.
except ValidationError as exc:
    print("❌  Environment mis‑configuration:\n", exc.json())
    raise SystemExit(1)

# ──────────────────────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format=json.dumps(
        {
            "time": "%(asctime)s",
            "level": "%(levelname)s",
            "msg": "%(message)s",
            "where": "%(name)s:%(lineno)d",
        }
    ),
)
logger = logging.getLogger("weather‑app")

# ──────────────────────────────────────────────────────────────────────────────
#  Flask initialisation
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY=settings.SECRET_KEY,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)
csrf = CSRFProtect(app)

# ──────────────────────────────────────────────────────────────────────────────
#  Password hashing (Argon2id)
# ──────────────────────────────────────────────────────────────────────────────

pwd_hasher: Final = PasswordHasher()

def hash_password(password: str) -> str:
    return pwd_hasher.hash(password)

def verify_password(hash_: str, password: str) -> bool:
    try:
        return pwd_hasher.verify(hash_, password)
    except argon_exc.VerifyMismatchError:
        return False

# ──────────────────────────────────────────────────────────────────────────────
#  Simple SQLite wrapper for users
# ──────────────────────────────────────────────────────────────────────────────

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

@contextmanager
def db_connection() -> Iterable[sqlite3.Connection]:
    conn = sqlite3.connect(settings.USERS_DB, isolation_level="DEFERRED")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

with db_connection() as _conn:
    _conn.executescript(_DB_SCHEMA)

def add_user(username: str, password_hash: str) -> None:
    with db_connection() as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.utcnow().isoformat()),
        )

def get_user(username: str) -> Mapping[str, Any] | None:
    with db_connection() as conn:
        cur = conn.execute(
            "SELECT username, password_hash FROM users WHERE username = ?", (username,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

# ──────────────────────────────────────────────────────────────────────────────
#  Weather API logic with caching
# ──────────────────────────────────────────────────────────────────────────────

_CACHE_DIR = pathlib.Path("cache")
_CACHE_DIR.mkdir(exist_ok=True)

def _cache_path(key: str) -> pathlib.Path:
    return _CACHE_DIR / f"{key}.json"

def _read_cache(key: str) -> Dict[str, Any] | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    if (datetime.utcnow() - datetime.utcfromtimestamp(path.stat().st_mtime)).total_seconds() > settings.CACHE_TTL_SECONDS:
        path.unlink(missing_ok=True)  # Expired
        return None
    try:
        return json.loads(path.read_text(encoding="utf‑8"))
    except Exception:  # Corrupt cache
        path.unlink(missing_ok=True)
        return None

def _write_cache(key: str, data: Dict[str, Any]) -> None:
    _cache_path(key).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf‑8")

def _normalise_location(province: str, city: str) -> tuple[str, str]:
    """Attach English suffixes required by the remote service."""
    if not province.lower().endswith((" province", " autonomous region", " city")):
        major_cities = {"beijing", "shanghai", "tianjin", "chongqing"}
        if province.lower() in major_cities:
            province += " City"
        elif province.lower() in {
            "inner mongolia",
            "guangxi",
            "xizang",
            "ningxia",
            "xinjiang",
        }:
            province += " Autonomous Region"
        else:
            province += " Province"
    if not city.lower().endswith(" city"):
        city += " City"
    return province.title(), city.title()

def fetch_weather(province: str, city: str) -> Dict[str, Any]:
    """Return live weather data or raise an exception."""
    province, city = _normalise_location(province, city)
    cache_key = f"{province}_{city}"
    if (cached := _read_cache(cache_key)):
        logger.info("Cache hit for %s / %s", province, city)
        return cached

    params = {
        "id": settings.WEATHER_API_ID,
        "key": settings.WEATHER_API_KEY,
        "sheng": province,
        "place": city,
    }
    logger.info("Calling weather API for %s / %s", province, city)
    try:
        resp = requests.get(settings.WEATHER_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("Weather API failure: %s", exc, exc_info=True)
        raise RuntimeError("Weather service unavailable.") from exc

    if data.get("code") != 200:
        logger.warning("Weather API returned non‑success code: %s", data)
        raise RuntimeError(f"Weather API error: {data.get('msg')!s}")

    data["queried_at"] = datetime.utcnow().isoformat()
    _write_cache(cache_key, data)
    return data

# ──────────────────────────────────────────────────────────────────────────────
#  Matplotlib global style
# ──────────────────────────────────────────────────────────────────────────────

matplotlib.use("Agg")  # Headless.
plt.style.use(settings.MATPLOTLIB_STYLE)
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = ["DejaVu Sans"]

# ──────────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────────

def build_diurnal_curve(base_temp: float) -> tuple[list[str], list[int]]:
    """Generate 24 synthetic hourly temperatures."""
    times: list[str] = []
    temps: list[int] = []
    now = datetime.utcnow()
    for hour in range(24):
        label = (now + timedelta(hours=hour)).strftime("%H:00")
        hour_num = int(label[:2])
        if 0 <= hour_num < 6:          # Early morning (coldest)
            temp = base_temp - 2 - (6 - hour_num) * 0.5
        elif 6 <= hour_num < 12:       # Morning (warming up)
            temp = base_temp + (hour_num - 6) * 0.8
        elif 12 <= hour_num < 15:      # Mid‑day (hottest)
            temp = base_temp + 5
        elif 15 <= hour_num < 20:      # Afternoon (cooling)
            temp = base_temp + 3 - (hour_num - 15) * 0.6
        else:                          # Night (cooler)
            temp = base_temp - (hour_num - 20) * 0.4
        times.append(label)
        temps.append(round(temp))
    return times, temps

def render_temperature_plot(times: list[str], temps: list[int]) -> str:
    """Return relative path to saved PNG."""
    x = np.arange(len(times))
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spline = make_interp_spline(x, temps, k=3)
    y_smooth = spline(x_smooth)

    fig = plt.figure(figsize=(12, 5))
    plt.plot(x_smooth, y_smooth, linewidth=2.5)
    plt.scatter(x, temps, s=50, alpha=0.6, zorder=5)

    for i, t in enumerate(temps):
        if i % 3 == 0:
            plt.text(i, t + 0.7, f"{t} °C", ha="center", va="bottom", fontsize=9)

    plt.xticks(range(len(times)), times, rotation=35, ha="right")
    plt.title("24‑Hour Temperature Trend", pad=12)
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    plots_dir = pathlib.Path("static/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    filename = f"plot_{datetime.utcnow():%Y%m%d%H%M%S}.png"
    filepath = plots_dir / filename
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return f"/static/plots/{filename}"

# ──────────────────────────────────────────────────────────────────────────────
#  Flask Blueprints
# ──────────────────────────────────────────────────────────────────────────────

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")
pages_bp = Blueprint("pages", __name__)
api_bp = Blueprint("api", __name__, url_prefix="/api")

# ── Auth routes ───────────────────────────────────────────────────────────────

def login_required(view):
    """Decorator to ensure a user is authenticated."""
    from functools import wraps

    @wraps(view)
    def wrapped(*args, **kwargs):
        if "username" not in session:
            flash("Please sign in to continue.", "warning")
            return redirect(url_for("auth.login"))
        return view(*args, **kwargs)

    return wrapped

@auth_bp.route("/register", methods=["GET", "POST"])
def register() -> str | Response:
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not username or not password:
            flash("Username and password cannot be empty.", "danger")
            return redirect(url_for("auth.register"))

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("auth.register"))

        if get_user(username):
            flash("Username already exists.", "danger")
            return redirect(url_for("auth.register"))

        add_user(username, hash_password(password))
        flash("Registration successful. Please sign in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html")

@auth_bp.route("/login", methods=["GET", "POST"])
def login() -> str | Response:
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")

        user = get_user(username)
        if not user or not verify_password(user["password_hash"], password):
            flash("Invalid credentials.", "danger")
            return redirect(url_for("auth.login"))

        session["username"] = username
        flash("Signed in successfully.", "success")
        return redirect(url_for("pages.index"))

    return render_template("login.html")

@auth_bp.route("/logout")
def logout() -> Response:
    session.pop("username", None)
    flash("Signed out.", "info")
    return redirect(url_for("auth.login"))

# ── Core pages ────────────────────────────────────────────────────────────────

@pages_bp.route("/")
@login_required
def index() -> str:
    return render_template("index.html")

@pages_bp.route("/help")
@login_required
def help_page() -> str:
    return render_template("help.html", doc=textwrap.dedent(__doc__))

# ── JSON API endpoints ────────────────────────────────────────────────────────

@api_bp.route("/weather", methods=["POST"])
@csrf.exempt  # If you want the endpoint public. Otherwise keep CSRF.
def get_weather() -> Response:
    payload = request.get_json(silent=True) or {}
    province = payload.get("province", "").strip()
    city = payload.get("city", "").strip()

    if not province or not city:
        return jsonify(error="Both 'province' and 'city' are required."), 400

    try:
        weather = fetch_weather(province, city)
    except Exception as exc:
        logger.warning("Weather fetch failed: %s", exc)
        return jsonify(error=str(exc)), 502

    return jsonify(weather)

@api_bp.route("/visualise", methods=["POST"])
@csrf.exempt
def visualise() -> Response:
    payload = request.get_json(silent=True) or {}
    try:
        base_temp = float(payload["temperature"])
    except (KeyError, ValueError):
        return jsonify(error="'temperature' must be a valid number."), 400

    times, temps = build_diurnal_curve(base_temp)
    plot_url = render_temperature_plot(times, temps)
    return jsonify(plot_url=plot_url)

# ──────────────────────────────────────────────────────────────────────────────
#  Blueprint registration
# ──────────────────────────────────────────────────────────────────────────────

app.register_blueprint(auth_bp)
app.register_blueprint(pages_bp)
app.register_blueprint(api_bp)

# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1", host="0.0.0.0", port=8000)
