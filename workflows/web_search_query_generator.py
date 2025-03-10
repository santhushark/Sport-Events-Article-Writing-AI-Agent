from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing import Annotated, List, TypedDict
from operator import add


class InputState(TypedDict):
    article: str


class OutputState(TypedDict):
    web_search_query: str


class OverallState(InputState, OutputState):
    messages: Annotated[List[BaseMessage], add]


def create_web_search_query_generator_agent():
    model_query_generator = ChatOpenAI(model="gpt-4o-mini")

    async def generate_web_search_query(state: OverallState):
        local_messages = state.get("messages", [])
        human_message = HumanMessage(content=state["article"])
        local_messages.append(human_message)
        system_message = SystemMessage(
            content="You are a web search query generator agent. Generate a web search query to do web search about a sports event mentioned below. The query should be regarding the sports event summary."
        )
        response = await model_query_generator.ainvoke([system_message, human_message])
        # print("wsqg: " + response.content)
        state["web_search_query"] = response.content
        state["messages"] = local_messages + [response]
        # print(state["messages"])
        return state

    web_search_query_generator_graph = StateGraph(OverallState, input=InputState, output=OutputState)
    web_search_query_generator_graph.add_node("web_search_query_generator", generate_web_search_query)
    web_search_query_generator_graph.add_edge(START, "web_search_query_generator")
    web_search_query_generator_graph.add_edge("web_search_query_generator", END)

    return web_search_query_generator_graph.compile()
