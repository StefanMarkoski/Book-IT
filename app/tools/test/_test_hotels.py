from dotenv import load_dotenv
load_dotenv()

from app.tools.hotels_tool import HotelsApiTool

tool = HotelsApiTool()
data = tool.search_hotels(city="Barcelona", limit=5, min_rating=4,amenities=["wifi", "pool"])
print(data)