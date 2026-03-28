"""
ATOM -- Real-World Intelligence Hub.

JARVIS doesn't just control Tony's lab -- he knows what's happening
in the WORLD. Weather outside, news headlines, calendar events,
timezone awareness, seasonal context, and location intelligence.

This module connects ATOM to the real world through:

  1. WEATHER -- Current conditions and forecast via wttr.in (free, no API key)
  2. NEWS -- Top headlines via RSS feeds (no API key, offline-safe)
  3. CALENDAR -- Time-aware schedule intelligence (holidays, events)
  4. WORLD CLOCK -- Multi-timezone awareness
  5. LOCATION -- IP-based geolocation (free APIs)
  6. TEMPORAL -- Season, daylight, lunar phase, special dates
  7. ENVIRONMENT -- Sunrise/sunset estimates, weather-appropriate suggestions

Every source has an offline fallback. ATOM never breaks because
the internet is down -- it just knows less.

Contract:
    get_world_context() -> WorldContext    # full real-world picture
    get_weather() -> WeatherInfo          # current weather
    get_headlines() -> list[str]          # top news
    get_briefing() -> str                 # morning briefing text
    refresh() -> None                     # force refresh all sources

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

logger = logging.getLogger("atom.real_world")

_CACHE_FILE = Path("data/real_world_cache.json")
_WEATHER_TTL_S = 1800        # 30 minutes
_NEWS_TTL_S = 3600           # 1 hour
_LOCATION_TTL_S = 86400      # 24 hours
_REQUEST_TIMEOUT_S = 8


# ── Data Classes ─────────────────────────────────────────────────────

@dataclass
class WeatherInfo:
    """Current weather conditions."""
    temperature_c: float = 0.0
    feels_like_c: float = 0.0
    condition: str = "unknown"
    humidity_pct: int = 0
    wind_kph: float = 0.0
    location: str = ""
    last_updated: float = 0.0
    is_stale: bool = True

    def summary(self) -> str:
        if self.is_stale and self.condition == "unknown":
            return "Weather data unavailable."
        stale_tag = " (cached)" if self.is_stale else ""
        return (
            f"{self.condition}, {self.temperature_c:.0f}°C "
            f"(feels like {self.feels_like_c:.0f}°C), "
            f"humidity {self.humidity_pct}%, "
            f"wind {self.wind_kph:.0f} km/h"
            f"{' in ' + self.location if self.location else ''}"
            f"{stale_tag}"
        )


@dataclass
class LocationInfo:
    """IP-based geolocation."""
    city: str = ""
    region: str = ""
    country: str = ""
    timezone: str = ""
    lat: float = 0.0
    lon: float = 0.0
    last_updated: float = 0.0


@dataclass
class TemporalContext:
    """Rich time awareness beyond just clock time."""
    season: str = ""
    day_period: str = ""
    is_weekend: bool = False
    is_holiday: bool = False
    holiday_name: str = ""
    days_until_weekend: int = 0
    sunrise_approx: str = ""
    sunset_approx: str = ""
    moon_phase: str = ""
    special_note: str = ""


@dataclass
class WorldContext:
    """Complete real-world intelligence picture."""
    weather: WeatherInfo = field(default_factory=WeatherInfo)
    location: LocationInfo = field(default_factory=LocationInfo)
    temporal: TemporalContext = field(default_factory=TemporalContext)
    headlines: list[str] = field(default_factory=list)
    world_clocks: dict[str, str] = field(default_factory=dict)
    last_refresh: float = 0.0

    def quality_score(self) -> float:
        """How complete is our world picture? 0.0 to 1.0."""
        scores = [
            0.3 if not self.weather.is_stale else 0.0,
            0.2 if self.location.city else 0.0,
            0.2 if self.headlines else 0.0,
            0.2 if self.temporal.season else 0.1,
            0.1 if self.world_clocks else 0.0,
        ]
        return sum(scores)


# ── The Hub ──────────────────────────────────────────────────────────

class RealWorldIntelligence:
    """Connects ATOM to the real world.

    All network calls are async-friendly (run in executor) with
    aggressive caching and graceful offline fallback.
    """

    __slots__ = (
        "_config", "_weather", "_location", "_headlines",
        "_temporal", "_world_clocks",
        "_weather_ts", "_news_ts", "_location_ts",
        "_task", "_shutdown", "_refresh_interval",
        "_cache_dirty",
    )

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("real_world", {})
        self._config = cfg
        self._refresh_interval = cfg.get("refresh_interval_s", 900)

        self._weather = WeatherInfo()
        self._location = LocationInfo()
        self._headlines: list[str] = []
        self._temporal = TemporalContext()
        self._world_clocks: dict[str, str] = {}

        self._weather_ts: float = 0.0
        self._news_ts: float = 0.0
        self._location_ts: float = 0.0

        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()
        self._cache_dirty = False

        self._load_cache()
        self._update_temporal()

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        self._task = asyncio.ensure_future(self._refresh_loop())
        logger.info(
            "Real-world intelligence started (refresh every %ds)",
            self._refresh_interval,
        )

    def stop(self) -> None:
        self._shutdown.set()
        if self._task and not self._task.done():
            self._task.cancel()
        self.persist()

    async def _refresh_loop(self) -> None:
        await asyncio.sleep(5.0)
        await self._refresh_all_async()

        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=float(self._refresh_interval),
                )
                break
            except asyncio.TimeoutError:
                pass
            await self._refresh_all_async()

    async def _refresh_all_async(self) -> None:
        """Refresh all sources in executor to avoid blocking."""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._refresh_all)
        except Exception:
            logger.debug("Real-world refresh failed", exc_info=True)

    def _refresh_all(self) -> None:
        now = time.time()
        self._update_temporal()
        self._update_world_clocks()

        if now - self._location_ts > _LOCATION_TTL_S:
            self._fetch_location()

        if now - self._weather_ts > _WEATHER_TTL_S:
            self._fetch_weather()

        if now - self._news_ts > _NEWS_TTL_S:
            self._fetch_news()

        if self._cache_dirty:
            self.persist()

    # ── Weather ──────────────────────────────────────────────────────

    def _fetch_weather(self) -> None:
        """Fetch weather from wttr.in (free, no API key)."""
        city = self._location.city or self._config.get("city", "")
        url = f"https://wttr.in/{city}?format=j1" if city else "https://wttr.in/?format=j1"

        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "ATOM/1.0", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            current = data.get("current_condition", [{}])[0]
            area = data.get("nearest_area", [{}])[0]

            self._weather = WeatherInfo(
                temperature_c=float(current.get("temp_C", 0)),
                feels_like_c=float(current.get("FeelsLikeC", 0)),
                condition=current.get("weatherDesc", [{}])[0].get("value", "unknown"),
                humidity_pct=int(current.get("humidity", 0)),
                wind_kph=float(current.get("windspeedKmph", 0)),
                location=area.get("areaName", [{}])[0].get("value", city),
                last_updated=time.time(),
                is_stale=False,
            )
            self._weather_ts = time.time()
            self._cache_dirty = True
            logger.info("Weather updated: %s", self._weather.summary())

        except Exception:
            self._weather.is_stale = True
            logger.debug("Weather fetch failed", exc_info=True)

    # ── News ─────────────────────────────────────────────────────────

    _RSS_FEEDS = [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://feeds.reuters.com/reuters/topNews",
    ]

    def _fetch_news(self) -> None:
        """Fetch top headlines from RSS feeds (no API key needed)."""
        headlines: list[str] = []

        for feed_url in self._RSS_FEEDS:
            if len(headlines) >= 10:
                break
            try:
                req = urllib.request.Request(
                    feed_url, headers={"User-Agent": "ATOM/1.0"},
                )
                with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
                    xml_data = resp.read()

                root = ElementTree.fromstring(xml_data)
                for item in root.iter("item"):
                    title_el = item.find("title")
                    if title_el is not None and title_el.text:
                        title = title_el.text.strip()
                        if title and len(title) > 10 and title not in headlines:
                            headlines.append(title)
                            if len(headlines) >= 10:
                                break
            except Exception:
                logger.debug("RSS feed failed: %s", feed_url, exc_info=True)
                continue

        if headlines:
            self._headlines = headlines
            self._news_ts = time.time()
            self._cache_dirty = True
            logger.info("News updated: %d headlines", len(headlines))

    # ── Location ─────────────────────────────────────────────────────

    def _fetch_location(self) -> None:
        """Get approximate location from IP (free, no API key)."""
        try:
            req = urllib.request.Request(
                "http://ip-api.com/json/?fields=city,regionName,country,timezone,lat,lon",
                headers={"User-Agent": "ATOM/1.0"},
            )
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            self._location = LocationInfo(
                city=data.get("city", ""),
                region=data.get("regionName", ""),
                country=data.get("country", ""),
                timezone=data.get("timezone", ""),
                lat=float(data.get("lat", 0)),
                lon=float(data.get("lon", 0)),
                last_updated=time.time(),
            )
            self._location_ts = time.time()
            self._cache_dirty = True
            logger.info(
                "Location updated: %s, %s, %s",
                self._location.city, self._location.region, self._location.country,
            )
        except Exception:
            logger.debug("Location fetch failed", exc_info=True)

    # ── Temporal Intelligence ────────────────────────────────────────

    _HOLIDAYS_INDIA = {
        (1, 1): "New Year's Day",
        (1, 26): "Republic Day",
        (8, 15): "Independence Day",
        (10, 2): "Gandhi Jayanti",
        (12, 25): "Christmas",
    }

    def _update_temporal(self) -> None:
        now = datetime.now()
        month = now.month
        hour = now.hour
        weekday = now.weekday()

        if month in (3, 4, 5):
            season = "spring"
        elif month in (6, 7, 8):
            season = "summer"
        elif month in (9, 10, 11):
            season = "autumn"
        else:
            season = "winter"

        if 5 <= hour < 8:
            period = "early_morning"
        elif 8 <= hour < 12:
            period = "morning"
        elif 12 <= hour < 14:
            period = "midday"
        elif 14 <= hour < 17:
            period = "afternoon"
        elif 17 <= hour < 20:
            period = "evening"
        elif 20 <= hour < 23:
            period = "night"
        else:
            period = "late_night"

        is_weekend = weekday >= 5
        days_to_weekend = max(0, 4 - weekday) if weekday < 5 else 0

        day_key = (now.month, now.day)
        is_holiday = day_key in self._HOLIDAYS_INDIA
        holiday_name = self._HOLIDAYS_INDIA.get(day_key, "")

        sunrise, sunset = self._estimate_sun_times(now)

        moon = self._estimate_moon_phase(now)

        special = ""
        if now.month == now.day:
            special = f"Fun fact: today's date is a repeating digit ({now.month}/{now.day})."
        if now.month == 12 and now.day == 31:
            special = "Last day of the year. Time for reflection."

        self._temporal = TemporalContext(
            season=season,
            day_period=period,
            is_weekend=is_weekend,
            is_holiday=is_holiday,
            holiday_name=holiday_name,
            days_until_weekend=days_to_weekend,
            sunrise_approx=sunrise,
            sunset_approx=sunset,
            moon_phase=moon,
            special_note=special,
        )

    @staticmethod
    def _estimate_sun_times(now: datetime) -> tuple[str, str]:
        """Rough sunrise/sunset estimate based on month (for ~20-30°N latitude)."""
        month_sunrise = {
            1: "06:55", 2: "06:45", 3: "06:20", 4: "05:55",
            5: "05:35", 6: "05:25", 7: "05:35", 8: "05:50",
            9: "06:05", 10: "06:15", 11: "06:35", 12: "06:50",
        }
        month_sunset = {
            1: "17:35", 2: "18:00", 3: "18:20", 4: "18:35",
            5: "18:55", 6: "19:10", 7: "19:10", 8: "18:50",
            9: "18:20", 10: "17:50", 11: "17:25", 12: "17:20",
        }
        return (
            month_sunrise.get(now.month, "06:00"),
            month_sunset.get(now.month, "18:00"),
        )

    @staticmethod
    def _estimate_moon_phase(now: datetime) -> str:
        """Approximate lunar phase using a known new moon reference."""
        ref_new_moon = datetime(2024, 1, 11)
        days_since = (now - ref_new_moon).days
        cycle_day = days_since % 29.53
        if cycle_day < 1.85:
            return "new_moon"
        if cycle_day < 7.38:
            return "waxing_crescent"
        if cycle_day < 9.23:
            return "first_quarter"
        if cycle_day < 14.77:
            return "waxing_gibbous"
        if cycle_day < 16.61:
            return "full_moon"
        if cycle_day < 22.15:
            return "waning_gibbous"
        if cycle_day < 23.99:
            return "last_quarter"
        return "waning_crescent"

    # ── World Clocks ─────────────────────────────────────────────────

    def _update_world_clocks(self) -> None:
        """Show time in key world cities."""
        now = datetime.utcnow()
        offsets = {
            "India (IST)": 5.5,
            "New York (EST)": -5,
            "London (GMT)": 0,
            "Tokyo (JST)": 9,
            "San Francisco (PST)": -8,
            "Dubai (GST)": 4,
        }
        clocks: dict[str, str] = {}
        for city, offset in offsets.items():
            city_time = now + timedelta(hours=offset)
            clocks[city] = city_time.strftime("%I:%M %p")
        self._world_clocks = clocks

    # ── Public API ───────────────────────────────────────────────────

    def get_world_context(self) -> WorldContext:
        """Get the complete real-world intelligence picture."""
        self._update_temporal()
        return WorldContext(
            weather=self._weather,
            location=self._location,
            temporal=self._temporal,
            headlines=self._headlines[:10],
            world_clocks=self._world_clocks,
            last_refresh=max(self._weather_ts, self._news_ts),
        )

    def get_weather_summary(self) -> str:
        return self._weather.summary()

    def get_news_summary(self, count: int = 5) -> str:
        if not self._headlines:
            return "No news headlines available right now."
        items = self._headlines[:count]
        numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(items))
        return f"Top headlines:\n{numbered}"

    def get_world_clock_summary(self) -> str:
        if not self._world_clocks:
            self._update_world_clocks()
        lines = [f"  {city}: {t}" for city, t in self._world_clocks.items()]
        return "World time:\n" + "\n".join(lines)

    def get_temporal_summary(self) -> str:
        t = self._temporal
        parts = [f"Season: {t.season}. Time of day: {t.day_period}."]
        if t.is_weekend:
            parts.append("It's the weekend.")
        elif t.days_until_weekend == 1:
            parts.append("Weekend starts tomorrow.")
        elif t.days_until_weekend > 0:
            parts.append(f"{t.days_until_weekend} days until the weekend.")
        if t.is_holiday:
            parts.append(f"Today is {t.holiday_name}.")
        parts.append(f"Sunrise: ~{t.sunrise_approx}. Sunset: ~{t.sunset_approx}.")
        parts.append(f"Moon phase: {t.moon_phase.replace('_', ' ')}.")
        if t.special_note:
            parts.append(t.special_note)
        return " ".join(parts)

    def get_briefing(self) -> str:
        """Generate a comprehensive morning/greeting briefing."""
        now = datetime.now()
        hour = now.hour

        if 5 <= hour < 12:
            greeting = "Good morning, Boss. Here's your briefing."
        elif 12 <= hour < 17:
            greeting = "Good afternoon, Boss. Quick update."
        elif 17 <= hour < 21:
            greeting = "Good evening, Boss. Here's what's happening."
        else:
            greeting = "Late night check-in, Boss."

        parts = [greeting]

        parts.append(f"It's {now.strftime('%A, %B %d')}. {self._temporal.season.title()} season.")

        if self._temporal.is_holiday:
            parts.append(f"Today is {self._temporal.holiday_name}.")

        if not self._weather.is_stale:
            parts.append(f"Weather: {self._weather.summary()}.")
            if self._weather.temperature_c > 35:
                parts.append("It's hot out there. Stay hydrated.")
            elif self._weather.temperature_c < 10:
                parts.append("Bundle up, it's cold outside.")
            if "rain" in self._weather.condition.lower():
                parts.append("Looks like rain. Take an umbrella if you're heading out.")

        if self._headlines:
            parts.append(f"Top news: {self._headlines[0]}.")
            if len(self._headlines) > 1:
                parts.append(f"Also: {self._headlines[1]}.")

        if self._temporal.days_until_weekend == 1:
            parts.append("Almost the weekend.")

        return " ".join(parts)

    def get_llm_context_block(self) -> str:
        """Compact real-world context for LLM prompt injection."""
        lines: list[str] = []
        t = self._temporal

        lines.append(
            f"[WORLD] {t.season.title()}, {t.day_period.replace('_', ' ')} | "
            f"{'Weekend' if t.is_weekend else 'Weekday'}"
        )

        if not self._weather.is_stale:
            lines.append(
                f"[WEATHER] {self._weather.condition}, "
                f"{self._weather.temperature_c:.0f}°C, "
                f"{self._weather.location}"
            )

        if self._location.city:
            lines.append(
                f"[LOCATION] {self._location.city}, {self._location.country}"
            )

        if t.is_holiday:
            lines.append(f"[TODAY] {t.holiday_name}")

        if self._headlines:
            lines.append(f"[NEWS] {self._headlines[0][:80]}")

        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────────

    def persist(self) -> None:
        try:
            _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "weather": {
                    "temperature_c": self._weather.temperature_c,
                    "feels_like_c": self._weather.feels_like_c,
                    "condition": self._weather.condition,
                    "humidity_pct": self._weather.humidity_pct,
                    "wind_kph": self._weather.wind_kph,
                    "location": self._weather.location,
                    "last_updated": self._weather.last_updated,
                },
                "location": {
                    "city": self._location.city,
                    "region": self._location.region,
                    "country": self._location.country,
                    "timezone": self._location.timezone,
                    "lat": self._location.lat,
                    "lon": self._location.lon,
                },
                "headlines": self._headlines[:10],
                "weather_ts": self._weather_ts,
                "news_ts": self._news_ts,
                "location_ts": self._location_ts,
            }
            _CACHE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self._cache_dirty = False
        except Exception:
            logger.debug("Real-world cache save failed", exc_info=True)

    def _load_cache(self) -> None:
        if not _CACHE_FILE.exists():
            return
        try:
            data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))

            w = data.get("weather", {})
            self._weather = WeatherInfo(
                temperature_c=w.get("temperature_c", 0),
                feels_like_c=w.get("feels_like_c", 0),
                condition=w.get("condition", "unknown"),
                humidity_pct=w.get("humidity_pct", 0),
                wind_kph=w.get("wind_kph", 0),
                location=w.get("location", ""),
                last_updated=w.get("last_updated", 0),
                is_stale=True,
            )

            loc = data.get("location", {})
            self._location = LocationInfo(
                city=loc.get("city", ""),
                region=loc.get("region", ""),
                country=loc.get("country", ""),
                timezone=loc.get("timezone", ""),
                lat=loc.get("lat", 0),
                lon=loc.get("lon", 0),
            )

            self._headlines = data.get("headlines", [])
            self._weather_ts = data.get("weather_ts", 0)
            self._news_ts = data.get("news_ts", 0)
            self._location_ts = data.get("location_ts", 0)

            logger.info("Real-world cache loaded (weather: %s, headlines: %d)",
                        self._weather.location or "unknown", len(self._headlines))
        except Exception:
            logger.debug("Real-world cache load failed", exc_info=True)

    def shutdown(self) -> None:
        self.stop()
        logger.info("Real-world intelligence shut down")
