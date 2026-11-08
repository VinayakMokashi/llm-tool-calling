"""
MCP server with two tools: current time and current weather (FastMCP).

FastMCP turns ordinary Python functions into MCP tools. The @mcp.tool()
decorator reads each function's name, type hints and docstring and publishes
them as the tool's name, input schema and description, so no JSON schema is
written by hand. The docstring is what the model reads when it decides how to
call the tool, which is why get_weather's docstring asks for official city
names: the geocoder finds "Bengaluru" but not always "Bangalore".

- get_current_time(timezone) - the time in an IANA timezone such as "Asia/Kolkata"
- get_weather(city)          - current conditions from Open-Meteo (free, no API key)

How to run
----------
Normally an MCP client starts this server for you (see client.py). To start it
yourself, run `python 04_mcp/server.py`: it then waits for MCP messages on
stdin and writes replies to stdout, so it prints nothing else.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from fastmcp import FastMCP

mcp = FastMCP("Time and Weather")

# WMO weather interpretation codes used by Open-Meteo
WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}


@mcp.tool()
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current time in a given IANA timezone (e.g. 'Asia/Tokyo', 'Asia/Kolkata')."""
    try:
        now = datetime.now(ZoneInfo(timezone))
        return f"Current time in {timezone}: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    except Exception as e:
        return f"Unknown timezone '{timezone}': {e}"


@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city using Open-Meteo (no API key required).
    Use the city's official name, e.g. 'Bengaluru' rather than 'Bangalore'."""
    try:
        # Step 1: city name -> latitude and longitude
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        geo_resp.raise_for_status()
        results = geo_resp.json().get("results")
        if not results:
            return f"City '{city}' not found."
        loc = results[0]
        resolved_name = ", ".join(filter(None, [loc.get("name", city), loc.get("admin1"), loc.get("country")]))

        # Step 2: coordinates -> current conditions
        wx_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,weather_code",
                "wind_speed_unit": "kmh",
                "timezone": "auto",
            },
            timeout=10,
        )
        wx_resp.raise_for_status()
        cur = wx_resp.json()["current"]
        desc = WMO_CODES.get(cur["weather_code"], f"Weather code {cur['weather_code']}")
        return (
            f"Weather in {resolved_name}: {desc}, "
            f"{cur['temperature_2m']}°C (feels like {cur['apparent_temperature']}°C), "
            f"Humidity: {cur['relative_humidity_2m']}%, Wind: {cur['wind_speed_10m']} km/h"
        )
    except Exception as e:
        # Return the problem as text, so the model can tell the user instead of the server failing
        return f"Could not get weather for {city}: {e}"


if __name__ == "__main__":
    mcp.run(show_banner=False)  # stdio transport: stdout is reserved for MCP messages
