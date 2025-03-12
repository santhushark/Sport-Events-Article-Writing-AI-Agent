from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


class InputState(TypedDict):
    web_search_result: str


class OutputState(TypedDict):
    agent_output: str


class OverallState(InputState, OutputState):
    pass


def create_article_writer_agent():
    model_article_writer = ChatOpenAI(model="gpt-4o-mini")

    async def write_article(state: OverallState):
        human_message = HumanMessage(content=state["web_search_result"])
        system_message = SystemMessage(
            content="Expand the following text to be at least 100 words. Maintain the original meaning while adding detail. Treat the original text as credible source. Just expand the text, no interpretation or anything else!"
        )
        response = await model_article_writer.ainvoke([system_message, human_message])
        state["agent_output"] = response.content
        return state

    article_writer_graph = StateGraph(OverallState, input=InputState, output=OutputState)
    article_writer_graph.add_node("write_article", write_article)
    article_writer_graph.add_edge(START, "write_article")
    article_writer_graph.add_edge("write_article", END)

    return article_writer_graph.compile()
