from operator import add
from typing import Annotated, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient


# Load environment variables
load_dotenv()


class InputState(TypedDict):
    article: str


class OutputState(TypedDict):
    agent_output: str


class OverallState(InputState, OutputState):
    messages: Annotated[List[BaseMessage], add]


@tool
def get_web_search_results(web_search_query: str):
    """Get Web Search results"""
    client = TavilyClient()
    res = client.search(web_search_query, search_depth="advanced", topic = "news", days= 4, max_results= 5, include_answer=True, include_raw_content=True)
    search_res_content = ""
    search_res_content+= "web_search_answer_summary: "+ res["answer"]
    for i in range(5):
        search_res_content+= f"web_search_source-{i+1}: " + res["results"][i]["content"] + "\n"
    
    return search_res_content


def create_web_search_agent():
    tools_web_search = [get_web_search_results]
    sport_event_info = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools_web_search)


    async def call_sport_event_web_search_tool(state: OverallState):
        local_messages = state.get("messages", [])
        if not local_messages:
            human_message = HumanMessage(content=state["article"])
            local_messages.append(human_message)

        system_message = SystemMessage(
            content="""You are an agent tasked with fetching information about a sports event.
            If the information about the sports event is available, return it. Otherwise, return 'Sports event information not available.'"""
        )

        response = await sport_event_info.ainvoke([system_message] + local_messages)

        state["agent_output"] = response.content
        state["messages"] = local_messages + [response]

        return state

    def should_continue(state: OverallState) -> Literal["tools", END]: # type: ignore
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    sport_event_info_graph = StateGraph(OverallState, input=InputState, output=OutputState)
    sport_event_info_graph.add_node("call_sport_event_web_search_tool", call_sport_event_web_search_tool)
    sport_event_info_graph.add_node("tools", ToolNode(tools_web_search))
    sport_event_info_graph.add_edge(START, "call_sport_event_web_search_tool")
    sport_event_info_graph.add_conditional_edges("call_sport_event_web_search_tool", should_continue)
    sport_event_info_graph.add_edge("tools", "call_sport_event_web_search_tool")

    return sport_event_info_graph.compile()
