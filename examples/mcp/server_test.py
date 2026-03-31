from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from server import ACTIVE_ENV_PATH, build_server as build_base_server, get_settings


OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_CODE_DESCRIPTIONS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def _weather_code_description(code: int | None) -> str | None:
    if code is None:
        return None
    return WEATHER_CODE_DESCRIPTIONS.get(code, f"unknown weather code {code}")


def _resolve_location(location: str) -> dict[str, Any]:
    query = location.strip()
    if not query:
        raise ValueError("location must not be empty")

    params = {
        "name": query,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    with httpx.Client(timeout=20, follow_redirects=True) as client:
        response = client.get(OPEN_METEO_GEOCODING_URL, params=params)
        response.raise_for_status()
        payload = response.json()

    results = payload.get("results") or []
    if not results:
        raise RuntimeError(f"No location match found for {location!r}.")

    match = results[0]
    admin_parts = [
        part
        for part in (
            match.get("admin1"),
            match.get("country"),
        )
        if isinstance(part, str) and part.strip()
    ]
    resolved_name = ", ".join([match.get("name", query), *admin_parts])
    return {
        "query": query,
        "resolved_name": resolved_name,
        "latitude": match["latitude"],
        "longitude": match["longitude"],
        "country": match.get("country"),
        "admin1": match.get("admin1"),
        "timezone": match.get("timezone"),
    }


def _fetch_weather(location: dict[str, Any], temperature_unit: Literal["celsius", "fahrenheit"]) -> dict[str, Any]:
    wind_speed_unit = "mph" if temperature_unit == "fahrenheit" else "kmh"
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "current": (
            "temperature_2m,apparent_temperature,relative_humidity_2m,"
            "precipitation,weather_code,wind_speed_10m"
        ),
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "forecast_days": 1,
        "temperature_unit": temperature_unit,
        "wind_speed_unit": wind_speed_unit,
        "precipitation_unit": "mm",
        "timezone": "auto",
    }

    with httpx.Client(timeout=20, follow_redirects=True) as client:
        response = client.get(OPEN_METEO_FORECAST_URL, params=params)
        response.raise_for_status()
        payload = response.json()

    current = payload.get("current") or {}
    daily = payload.get("daily") or {}
    current_units = payload.get("current_units") or {}
    daily_units = payload.get("daily_units") or {}
    daily_weather_codes = daily.get("weather_code") or []
    daily_max_temps = daily.get("temperature_2m_max") or []
    daily_min_temps = daily.get("temperature_2m_min") or []
    daily_precip_probs = daily.get("precipitation_probability_max") or []

    return {
        "location_query": location["query"],
        "resolved_location": location["resolved_name"],
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "timezone": payload.get("timezone") or location.get("timezone"),
        "temperature_unit": temperature_unit,
        "current": {
            "time": current.get("time"),
            "temperature": current.get("temperature_2m"),
            "temperature_unit": current_units.get("temperature_2m"),
            "apparent_temperature": current.get("apparent_temperature"),
            "relative_humidity": current.get("relative_humidity_2m"),
            "relative_humidity_unit": current_units.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_speed_unit": current_units.get("wind_speed_10m"),
            "precipitation": current.get("precipitation"),
            "precipitation_unit": current_units.get("precipitation"),
            "weather_code": current.get("weather_code"),
            "condition": _weather_code_description(current.get("weather_code")),
        },
        "today_forecast": {
            "date": (daily.get("time") or [None])[0],
            "max_temperature": daily_max_temps[0] if daily_max_temps else None,
            "min_temperature": daily_min_temps[0] if daily_min_temps else None,
            "temperature_unit": daily_units.get("temperature_2m_max"),
            "max_precipitation_probability": daily_precip_probs[0] if daily_precip_probs else None,
            "precipitation_probability_unit": daily_units.get("precipitation_probability_max"),
            "weather_code": daily_weather_codes[0] if daily_weather_codes else None,
            "condition": _weather_code_description(daily_weather_codes[0] if daily_weather_codes else None),
        },
        "source": "Open-Meteo APIs (free, no API key required)",
    }


def build_server(host: str, port: int, path: str):
    server = build_base_server(host=host, port=port, path=path)

    @server.tool(
        name="get_current_time",
        description=(
            "Returns the current server time. Useful for testing MCP tool calling without any paid API."
        ),
    )
    def get_current_time(timezone_name: str = "UTC") -> str:
        try:
            tz = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown timezone: {timezone_name}") from exc

        now_utc = datetime.now(timezone.utc)
        now_local = now_utc.astimezone(tz)
        result = {
            "timezone": timezone_name,
            "utc_time": now_utc.isoformat(),
            "local_time": now_local.isoformat(),
            "weekday": now_local.strftime("%A"),
            "source": "local server clock",
        }
        return json.dumps(result, ensure_ascii=False)

    @server.tool(
        name="get_weather",
        description=(
            "Fetches current weather and today's forecast from Open-Meteo. "
            "This uses a free public API and does not require any API key."
        ),
    )
    def get_weather(
        location: str,
        temperature_unit: Literal["celsius", "fahrenheit"] = "celsius",
    ) -> str:
        resolved_location = _resolve_location(location)
        result = _fetch_weather(resolved_location, temperature_unit=temperature_unit)
        return json.dumps(result, ensure_ascii=False)

    return server


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Run the MCP example server with extra free tools for safe tool-calling tests."
    )
    parser.add_argument("--host", default=settings.mcp_host)
    parser.add_argument("--port", type=int, default=settings.mcp_port)
    parser.add_argument("--path", default=settings.mcp_path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = build_server(host=args.host, port=args.port, path=args.path)
    print(f"Test MCP server listening on http://{args.host}:{args.port}{args.path}")
    print("Free tools available: get_current_time, get_weather")
    print(f"Loaded env from {ACTIVE_ENV_PATH}")
    try:
        server.run(transport="streamable-http")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
