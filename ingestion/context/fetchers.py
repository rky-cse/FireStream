import os
import datetime
import requests
import pytz
import holidays
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# Environment configuration
OWM_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
FESTIVAL_COUNTRY = os.getenv("FESTIVAL_COUNTRY", "IN")  # default to India


def fetch_local_time(timezone_str: str = None) -> dict:
    """
    Returns local date, time, and a weekend flag.
    If timezone_str is None, uses system local timezone.
    """
    tz = pytz.timezone(timezone_str) if timezone_str else datetime.datetime.now().astimezone().tzinfo
    now = datetime.datetime.now(tz)
    return {
        "date": now.date().isoformat(),
        "time": now.time().strftime("%H:%M:%S"),
        "is_weekend": now.weekday() >= 5,
    }


def fetch_weather(lat: float, lon: float) -> dict:
    """
    Fetches current weather conditions from OpenWeatherMap using the API key.
    Returns a dict with keys: main, description, temperature, humidity, wind_speed.
    Raises:
      - OSError if the API key is missing.
      - RuntimeError with status code details on HTTP errors.
    """
    if not OWM_API_KEY:
        raise OSError("OPENWEATHERMAP_API_KEY environment variable not set. Please add it to your .env file or environment.")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OWM_API_KEY, "units": "metric"}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 401:
            raise RuntimeError("Unauthorized: Invalid OpenWeatherMap API key (HTTP 401). Check your API key and ensure it's active.")
        else:
            raise RuntimeError(f"HTTP error occurred: {http_err} (status code {response.status_code})")
    except requests.exceptions.RequestException as req_err:
        raise RuntimeError(f"Error connecting to OpenWeatherMap: {req_err}")

    data = response.json()
    # Parse and return relevant fields
    return {
        "main": data["weather"][0]["main"],
        "description": data["weather"][0]["description"],
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"],
    }


def fetch_festivals(country: str = FESTIVAL_COUNTRY, year: int = None) -> list:
    """
    Returns a list of festival/holiday dictionaries for the given country and year.
    Uses python-holidays for holiday lookups.
    """
    if not year:
        year = datetime.datetime.now().year
    try:
        country_holidays = holidays.CountryHoliday(country, years=[year])
    except Exception as e:
        raise ValueError(f"Error fetching holidays for country '{country}': {e}")

    return [{"date": date.isoformat(), "name": name} for date, name in country_holidays.items()]


if __name__ == '__main__':
    # Example usage
    ctx = fetch_local_time('Asia/Kolkata')
    print("Context Time:", ctx)

    try:
        weather = fetch_weather(22.5726, 88.3639)
        print("Weather:", weather)
    except Exception as e:
        print("Failed to fetch weather:", e)

    try:
        festivals = fetch_festivals()
        print("Festivals:", festivals)
    except Exception as e:
        print("Failed to fetch festivals:", e)
