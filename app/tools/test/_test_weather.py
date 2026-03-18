from dotenv import load_dotenv
load_dotenv()

from app.tools.weather_tool import OpenWeatherTool

tool = OpenWeatherTool()
print(tool.get_forecast_5day("Athens"))