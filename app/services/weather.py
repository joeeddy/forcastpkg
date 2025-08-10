"""Weather data service using Open-Meteo API (no API key required)."""

import httpx
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import asyncio

from app.models import WeatherData

class WeatherService:
    """Service for fetching weather data from Open-Meteo API."""
    
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1"
        self.geocoding_url = "https://geocoding-api.open-meteo.com/v1"
    
    async def get_coordinates(self, location: str) -> tuple[float, float]:
        """Get latitude and longitude for a location."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.geocoding_url}/search",
                    params={'name': location, 'count': 1, 'language': 'en', 'format': 'json'}
                )
                response.raise_for_status()
                data = response.json()
            
            if not data.get('results'):
                raise ValueError(f"Location '{location}' not found")
            
            result = data['results'][0]
            return result['latitude'], result['longitude']
        except Exception as e:
            raise ValueError(f"Error getting coordinates for {location}: {str(e)}")
    
    async def get_current_weather(self, location: str) -> WeatherData:
        """Get current weather data for a location."""
        try:
            lat, lon = await self.get_coordinates(location)
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/forecast",
                    params={
                        'latitude': lat,
                        'longitude': lon,
                        'current_weather': 'true',
                        'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m',
                        'timezone': 'auto'
                    }
                )
                response.raise_for_status()
                data = response.json()
            
            current = data.get('current_weather', {})
            hourly = data.get('hourly', {})
            
            # Get current hour's detailed data
            current_time = datetime.now()
            hour_index = 0  # Use first available hour as current
            
            humidity = None
            wind_speed = current.get('windspeed')
            
            if hourly.get('relative_humidity_2m'):
                humidity = hourly['relative_humidity_2m'][hour_index]
            
            # Map weather codes to descriptions
            weather_code = current.get('weathercode', 0)
            weather_description = self._get_weather_description(weather_code)
            
            return WeatherData(
                temperature=current.get('temperature', 0),
                humidity=humidity,
                wind_speed=wind_speed,
                weather_description=weather_description,
                timestamp=datetime.now(),
                location=location
            )
        except Exception as e:
            raise ValueError(f"Error fetching weather data for {location}: {str(e)}")
    
    async def get_weather_forecast(self, location: str, days: int = 7) -> List[WeatherData]:
        """Get weather forecast for multiple days."""
        try:
            lat, lon = await self.get_coordinates(location)
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/forecast",
                    params={
                        'latitude': lat,
                        'longitude': lon,
                        'daily': 'temperature_2m_max,temperature_2m_min,weathercode,windspeed_10m_max,relative_humidity_2m_mean',
                        'timezone': 'auto',
                        'forecast_days': min(days, 14)  # Open-Meteo supports up to 14 days
                    }
                )
                response.raise_for_status()
                data = response.json()
            
            daily = data.get('daily', {})
            forecast_data = []
            
            if daily.get('time'):
                for i, date_str in enumerate(daily['time']):
                    try:
                        date = datetime.fromisoformat(date_str)
                        temp_max = daily.get('temperature_2m_max', [])[i] if i < len(daily.get('temperature_2m_max', [])) else None
                        temp_min = daily.get('temperature_2m_min', [])[i] if i < len(daily.get('temperature_2m_min', [])) else None
                        weather_code = daily.get('weathercode', [])[i] if i < len(daily.get('weathercode', [])) else 0
                        wind_speed = daily.get('windspeed_10m_max', [])[i] if i < len(daily.get('windspeed_10m_max', [])) else None
                        humidity = daily.get('relative_humidity_2m_mean', [])[i] if i < len(daily.get('relative_humidity_2m_mean', [])) else None
                        
                        # Use average of max and min temperature
                        avg_temp = (temp_max + temp_min) / 2 if temp_max and temp_min else (temp_max or temp_min or 0)
                        
                        weather_data = WeatherData(
                            temperature=avg_temp,
                            humidity=humidity,
                            wind_speed=wind_speed,
                            weather_description=self._get_weather_description(weather_code),
                            timestamp=date,
                            location=location
                        )
                        forecast_data.append(weather_data)
                    except Exception:
                        continue  # Skip invalid data points
            
            return forecast_data
        except Exception as e:
            raise ValueError(f"Error fetching weather forecast for {location}: {str(e)}")
    
    async def get_multiple_cities_weather(self, cities: List[str]) -> Dict[str, WeatherData]:
        """Get current weather for multiple cities."""
        try:
            weather_data = {}
            
            # Fetch weather data for all cities concurrently
            tasks = [self.get_current_weather(city) for city in cities]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for city, result in zip(cities, results):
                if isinstance(result, Exception):
                    continue  # Skip failed requests
                weather_data[city] = result
            
            return weather_data
        except Exception as e:
            raise ValueError(f"Error fetching weather for multiple cities: {str(e)}")
    
    def _get_weather_description(self, weather_code: int) -> str:
        """Convert Open-Meteo weather code to description."""
        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            56: "Light freezing drizzle",
            57: "Dense freezing drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            66: "Light freezing rain",
            67: "Heavy freezing rain",
            71: "Slight snow fall",
            73: "Moderate snow fall",
            75: "Heavy snow fall",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }
        return weather_codes.get(weather_code, f"Unknown weather (code: {weather_code})")
    
    async def get_weather_alerts(self, location: str) -> List[Dict]:
        """Get weather alerts/warnings for a location (placeholder implementation)."""
        # Open-Meteo doesn't provide alerts in the free tier
        # This is a placeholder that could be extended with other APIs
        try:
            # For demonstration, we'll check if current weather is extreme
            current_weather = await self.get_current_weather(location)
            alerts = []
            
            if current_weather.temperature > 35:  # Hot weather
                alerts.append({
                    "type": "heat_warning",
                    "message": f"High temperature alert: {current_weather.temperature}°C",
                    "severity": "moderate"
                })
            elif current_weather.temperature < -10:  # Cold weather
                alerts.append({
                    "type": "cold_warning", 
                    "message": f"Low temperature alert: {current_weather.temperature}°C",
                    "severity": "moderate"
                })
            
            if current_weather.wind_speed and current_weather.wind_speed > 25:  # High wind
                alerts.append({
                    "type": "wind_warning",
                    "message": f"High wind speed: {current_weather.wind_speed} km/h",
                    "severity": "moderate"
                })
            
            return alerts
        except Exception:
            return []  # Return empty list if unable to fetch alerts