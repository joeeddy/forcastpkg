"""Weather data API endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict

from app.services.weather import WeatherService
from app.models import WeatherData

router = APIRouter()
weather_service = WeatherService()

@router.get("/current/{location}", response_model=WeatherData)
async def get_current_weather(location: str):
    """Get current weather data for a location."""
    try:
        return await weather_service.get_current_weather(location)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/forecast/{location}")
async def get_weather_forecast(
    location: str,
    days: int = Query(7, ge=1, le=14, description="Number of days for forecast (max 14)")
):
    """Get weather forecast for a location."""
    try:
        forecast_data = await weather_service.get_weather_forecast(location, days)
        return {
            "location": location,
            "forecast_days": days,
            "forecast_data": [weather.dict() for weather in forecast_data]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/multiple")
async def get_multiple_cities_weather(
    cities: str = Query(..., description="Comma-separated list of cities")
):
    """Get current weather for multiple cities."""
    try:
        city_list = [city.strip() for city in cities.split(",")]
        
        if len(city_list) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 cities allowed")
        
        weather_data = await weather_service.get_multiple_cities_weather(city_list)
        
        return {
            "cities_requested": city_list,
            "weather_data": {city: weather.dict() for city, weather in weather_data.items()},
            "successful_fetches": len(weather_data)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/alerts/{location}")
async def get_weather_alerts(location: str):
    """Get weather alerts/warnings for a location."""
    try:
        alerts = await weather_service.get_weather_alerts(location)
        return {
            "location": location,
            "alerts": alerts,
            "alert_count": len(alerts)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/coordinates/{location}")
async def get_location_coordinates(location: str):
    """Get latitude and longitude coordinates for a location."""
    try:
        lat, lon = await weather_service.get_coordinates(location)
        return {
            "location": location,
            "latitude": lat,
            "longitude": lon
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/compare")
async def compare_weather(
    locations: str = Query(..., description="Comma-separated list of locations to compare")
):
    """Compare weather conditions across multiple locations."""
    try:
        location_list = [loc.strip() for loc in locations.split(",")]
        
        if len(location_list) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 locations allowed for comparison")
        
        weather_data = await weather_service.get_multiple_cities_weather(location_list)
        
        # Create comparison data
        comparison = {
            "comparison_data": {},
            "summary": {
                "warmest": None,
                "coldest": None,
                "windiest": None,
                "most_humid": None
            }
        }
        
        temperatures = []
        wind_speeds = []
        humidities = []
        
        for location, weather in weather_data.items():
            comparison["comparison_data"][location] = weather.dict()
            temperatures.append((location, weather.temperature))
            
            if weather.wind_speed:
                wind_speeds.append((location, weather.wind_speed))
            
            if weather.humidity:
                humidities.append((location, weather.humidity))
        
        # Find extremes
        if temperatures:
            comparison["summary"]["warmest"] = max(temperatures, key=lambda x: x[1])
            comparison["summary"]["coldest"] = min(temperatures, key=lambda x: x[1])
        
        if wind_speeds:
            comparison["summary"]["windiest"] = max(wind_speeds, key=lambda x: x[1])
        
        if humidities:
            comparison["summary"]["most_humid"] = max(humidities, key=lambda x: x[1])
        
        return comparison
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/popular-cities")
async def get_popular_cities_weather():
    """Get weather for popular cities around the world."""
    try:
        popular_cities = [
            "New York", "London", "Tokyo", "Paris", "Sydney",
            "Los Angeles", "Dubai", "Singapore", "Mumbai", "Berlin"
        ]
        
        weather_data = await weather_service.get_multiple_cities_weather(popular_cities)
        
        return {
            "popular_cities_weather": {city: weather.dict() for city, weather in weather_data.items()},
            "cities_count": len(weather_data),
            "timestamp": weather_data[list(weather_data.keys())[0]].timestamp if weather_data else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))