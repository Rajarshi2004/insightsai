from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os, io, base64
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import markdown
import requests
import google.generativeai as genai
from pydantic import BaseModel
import uuid
from xgboost import XGBRegressor
chat_sessions = {}
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

# ---------------- ENV ----------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
PIXABAY_KEY = os.getenv("PIXABAY_KEY")

# ---------------- APP ----------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

session_history = {}

# ---------------- SEARCH ----------------
def perform_web_search(query: str):
    if not SERPAPI_KEY:
        return "SERPAPI_KEY missing"
    url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}"
    res = requests.get(url).json()
    results = []
    for r in res.get("organic_results", [])[:3]:
        results.append(f"- [{r.get('title')}]({r.get('link')}): {r.get('snippet')}")
    return "\n".join(results)

# ---------------- PIXABAY ----------------
def get_image_from_pixabay(query: str):
    if not PIXABAY_KEY:
        return None
    url = f"https://pixabay.com/api/?key={PIXABAY_KEY}&q={query}&image_type=photo"
    res = requests.get(url).json()
    return [h["webformatURL"] for h in res.get("hits", [])[:3]]

def needs_web(msg: str):
    keys = ["latest", "news", "today", "price", "current", "who is", "what is","search"]
    return any(k in msg.lower() for k in keys)

def needs_image(msg: str):
    keys = ["image", "picture", "photo", "show me"]
    return any(k in msg.lower() for k in keys)
# ---------------- ROUTES ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/ml", response_class=HTMLResponse)
def ml_page(request: Request):
    return templates.TemplateResponse("ml.html", {"request": request})

# ---------------- CHAT ----------------
chat_sessions = {}

@app.post("/chat")
async def chat(prompt: str = Form(...), session_id: str = Form(None)):

    if not session_id or session_id not in chat_sessions:
        session_id = str(uuid.uuid4())
        chat = model.start_chat(history=[])
        chat_sessions[session_id] = chat
    else:
        chat = chat_sessions[session_id]

    tool_context = ""
    
    if needs_web(prompt):
        web_data = perform_web_search(prompt)
        tool_context += f"\nWeb Search Results:\n{web_data}\n"

    if needs_image(prompt):
        imgs = get_image_from_pixabay(prompt)
        image_html = "".join(f"<img src='{i}' width='45%'>" for i in imgs)
        tool_context += f"\nImage URLs:\n{imgs}\n"
    else:
        image_html = ""

    final_prompt = prompt
    if tool_context:
        final_prompt += f"\n\nUse this external data:\n{tool_context}"

    # THIS IS THE KEY LINE
    response = chat.send_message(final_prompt)

    reply = markdown.markdown(response.text) + image_html

    return JSONResponse({
        "reply": reply,
        "session_id": session_id
    })


# ---------------- ML FORECAST ----------------
@app.post("/ml/forecast")
def forecast_stock(data: dict):
    try:
        # ===== INPUT =====
        stock = str(data.get("stock", "")).upper().strip()
        horizon = int(data.get("horizon", 7))

        if not stock:
            raise HTTPException(status_code=400, detail="Stock symbol required")

        # ===== DOWNLOAD DATA =====
        df = yf.download(stock, period="2y", interval="1d", progress=False)

        if df.empty:
            raise HTTPException(status_code=400, detail="Invalid stock symbol")

        # ===== FIX: HANDLE MULTIINDEX FROM YFINANCE =====
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # ===== FEATURE ENGINEERING =====
        df["Return"] = df["Close"].pct_change()
        df["MA10"] = df["Close"].rolling(10).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["Volatility"] = df["Return"].rolling(10).std()

        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26

        df["Target"] = df["Return"].shift(-1)
        df.dropna(inplace=True)

        features = ["MA10", "MA20", "Volatility", "RSI", "MACD"]

        X = df[features]
        y = df["Target"]

        # ===== MODEL =====
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5
        )
        model.fit(X, y)

        # ===== FORECAST =====
        last_price = float(df["Close"].iloc[-1])
        last_features = X.iloc[-1:].values

        preds = []
        for _ in range(horizon):
            prediction = float(model.predict(last_features)[0])
            preds.append(prediction)

        future_prices = [last_price]
        for r in preds:
            future_prices.append(future_prices[-1] * (1 + r))

        future_prices = future_prices[1:]

        future_dates = pd.date_range(
            start=df.index[-1] + pd.Timedelta(days=1),
            periods=horizon,
            freq="B"
        )

        mean_series = pd.Series(future_prices, index=future_dates)

        std = float(df["Return"].std())
        lower_series = mean_series * (1 - std)
        upper_series = mean_series * (1 + std)

        trend = "Bullish" if future_prices[-1] > last_price else "Bearish"
        confidence = "High" if std < 0.02 else "Moderate"

        close_series = df["Close"].astype(float)

        return {
            "stock": stock,
            "last_close": float(last_price),

            "historical_dates": close_series.index[-120:].strftime("%Y-%m-%d").tolist(),
            "historical_prices": close_series.iloc[-120:].tolist(),

            "forecast_dates": future_dates.strftime("%Y-%m-%d").tolist(),
            "forecast_mean": mean_series.astype(float).tolist(),
            "lower_ci": lower_series.astype(float).tolist(),
            "upper_ci": upper_series.astype(float).tolist(),

            "trend": trend,
            "confidence": confidence
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )
