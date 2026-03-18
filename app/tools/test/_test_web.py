from dotenv import load_dotenv
load_dotenv()

from app.tools.web_search_tool import WebSearchTool


tool = WebSearchTool()
data = tool.search("What is the best time to visit Madrid?", max_results=3)
print(data)