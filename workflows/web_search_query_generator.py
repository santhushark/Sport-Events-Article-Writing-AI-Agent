from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing import Annotated, List, TypedDict
from operator import add


class InputState(TypedDict):
    event: str


class OutputState(TypedDict):
    agent_output: str


class OverallState(InputState, OutputState):
    pass


def create_web_search_query_generator_agent():
    model_query_generator = ChatOpenAI(model="gpt-4o-mini")

    async def generate_web_search_query(state: OverallState):
        human_message = HumanMessage(content=state["event"])
        system_message = SystemMessage(
            content="You are a web search query generator agent. Generate a web search query to do web search about a sports event mentioned below. The query should be regarding the sports event summary."
        )
        response = await model_query_generator.ainvoke([system_message, human_message])
        state["agent_output"] = response.content
        return state

    web_search_query_generator_graph = StateGraph(OverallState, input=InputState, output=OutputState)
    web_search_query_generator_graph.add_node("web_search_query_generator", generate_web_search_query)
    web_search_query_generator_graph.add_edge(START, "web_search_query_generator")
    web_search_query_generator_graph.add_edge("web_search_query_generator", END)

    return web_search_query_generator_graph.compile()
