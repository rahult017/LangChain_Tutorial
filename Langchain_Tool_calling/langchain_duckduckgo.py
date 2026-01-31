from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke("top news in india today")

print(results)
print("\n\n", search_tool.name)
print("\n\n", search_tool.description)
print("\n\n", search_tool.args)
