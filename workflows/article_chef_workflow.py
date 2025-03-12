from typing import Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .article_writer import create_article_writer_agent
from .web_search import create_web_search_agent
from .web_search_query_generator import create_web_search_query_generator_agent


class ArticlePostabilityGrader(BaseModel):
    """Binary scores for verifying if an article mentions sport name, team names, tournament name, and meets the minimum word count of 100 words."""

    ontopic: str = Field(
        description="The Article is about a sport event, 'yes' or 'no'"
    )
    sport_name_mentioned: str = Field(
        description="The article mentions the name of the sport, 'yes' or 'no'"
    )
    teams_mentioned: str = Field(
        description="The article mentions the 2 team names or 2 individual player names, 'yes' or 'no'"
    )
    tournament_name_mentioned: str = Field(
        description="The article mentions the name of the tournament, 'yes' or 'no'"
    )
    meets_100_words: str = Field(
        description="The article has at least 100 words, 'yes' or 'no'"
    )


class InputArticleState(TypedDict):
    event: str


class OutputFinalArticleState(TypedDict):
    final_article: str
    ontopic: str


class SharedArticleState(InputArticleState, OutputFinalArticleState):
    mentions_sport_name: str
    mentions_team_names: str
    mentions_tournament_name: str
    web_search_result: str
    web_search_query_generated: str
    meets_100_words: str


# Article Chef agent, Supervises web_search_query_generator, web_search and article_writer agent
class ArticleWorkflow:
    def __init__(self, llm_model="gpt-4o-mini", temperature=0):
        self.web_search_query_generator_agent = create_web_search_query_generator_agent()
        self.web_search_agent = create_web_search_agent()
        self.article_writer_agent = create_article_writer_agent()
        self.llm_postability = ChatOpenAI(model=llm_model, temperature=temperature)
        self.workflow = self._create_workflow()

    def _create_postability_grader(self):
        prompt_template = """
        You are a grader assessing whether a event information meets the following criteria:
        1. The event is about sports or not. If yes answer, answer with 'yes' for ontopic, otherwise with 'no'.
        2. The event explicitly mentions the name of the sport (e.g., "Cricket" or "Football" or "Hockey".etc.). If it is present answer with 'yes' for sport_name_mentioned, otherwise respond with 'no'.
        3. The event explicitly mentions the sport event details, for example, by stating "2 team names" for a team sport or "2 player names" for a individual sport (e.g., "India vs Pakistan" or "Roger Federer vs Rafael Nadal"). If this is present, respond with 'yes' for teams_mentioned; otherwise, respond 'no'.
        4. The event mentions the tournament name (e.g. "ICC Champions Trophy" or "World Cup" or "English Premier League" or "Asia Cup" or "Wimbledon" or "French Open"). If it is present answer with 'yes' for tournament_name_mentioned; otherwise answer with 'no'.
        5. The event contains at least 100 words. If this is met, respond with 'yes' for meets_100_words; otherwise, respond 'no'.

        Provide four binary scores ('yes' or 'no') as follows:
        - ontopic: 'yes' or 'no' depending on whether the article is related to a sport event.
        - sport_name_mentioned: 'yes' or 'no' depending on whether the article mentions the name of the sport.
        - teams_mentioned: 'yes' or 'no' depending on whether the article mentions the name of 2 players or 2 teams.
        - tournament_name_mentioned: 'yes' or 'no' depending on whether the article mentions the name of the tournament.
        - meets_100_words: 'yes' or 'no' depending on whether the article has at least 100 words.
        """
        postability_system = ChatPromptTemplate.from_messages(
            [("system", prompt_template), ("human", "Event:\n\n{event}")]
        )
        return postability_system | self.llm_postability.with_structured_output(
            ArticlePostabilityGrader
        )

    async def update_event_state(self, state: SharedArticleState) -> SharedArticleState:
        article_chef = self._create_postability_grader()
        states_to_check = ["ontopic", "mentions_sport_name", "mentions_team_names", "mentions_tournament_name", "meets_100_words"]
        if not all(key in state for key in states_to_check):
            response = await article_chef.ainvoke({"event": state["event"]})
            state["ontopic"] = response.ontopic
            state["mentions_sport_name"] = response.sport_name_mentioned
            state["mentions_team_names"] = response.teams_mentioned
            state["mentions_tournament_name"] = response.tournament_name_mentioned
            state["meets_100_words"] = response.meets_100_words

        return state

    # Web search query generator node, Calls Query Generator Agent
    async def web_search_query_gen_node(self, state: SharedArticleState) -> SharedArticleState:
        response = await self.web_search_query_generator_agent.ainvoke({"event": state["event"]})
        state["web_search_query_generated"] = f"{response['agent_output']}"
        return state

    # Web Search node, Calls Web Search Agent
    async def web_search_node(self, state: SharedArticleState) -> SharedArticleState:
        response = await self.web_search_agent.ainvoke({"web_search_query": state["web_search_query_generated"]})
        state["web_search_result"] = f"{response['agent_output']}"
        return state

    # Article writer mode, calls article writer agent
    async def article_writer_node(self, state: SharedArticleState) -> SharedArticleState:
        response = await self.article_writer_agent.ainvoke({"web_search_result": state["web_search_result"]})
        state["final_article"] = response["agent_output"]
        state["meets_100_words"] = "yes"
        return state

    # decides what agent to call next
    def article_chef_decider(self,state: SharedArticleState,) -> Literal["web_search_query_generator", "web_searcher", "article_writer", END]: # type: ignore
        if (
            state["ontopic"] == "no" 
            or state["mentions_sport_name"] == "no" 
            or state["mentions_team_names"] == "no" 
            or state["mentions_tournament_name"] == "no"
            ):
            return END
        elif "web_search_query_generated" not in state:
            next_node = "web_search_query_generator"
        elif "web_search_result" not in state:
            next_node = "web_searcher"
        elif state["meets_100_words"] == "no":
            next_node = "article_writer"
        else:
            next_node = END
        return next_node

    # Creating supervisor agent workflow
    def _create_workflow(self):
        workflow = StateGraph(
            SharedArticleState, input=InputArticleState, output=OutputFinalArticleState
        )
        workflow.add_node("article_chef", self.update_event_state)
        workflow.add_node("web_search_query_generator", self.web_search_query_gen_node)
        workflow.add_node("web_searcher", self.web_search_node)
        workflow.add_node("article_writer", self.article_writer_node)
        workflow.set_entry_point("article_chef")
        workflow.add_conditional_edges(
            "article_chef",
            self.article_chef_decider,
            {
                "web_search_query_generator": "web_search_query_generator",
                "web_searcher": "web_searcher",
                "article_writer": "article_writer",
                END: END,
            },
        )
        workflow.add_edge("web_searcher", "article_chef")
        workflow.add_edge("article_writer", "article_chef")
        workflow.add_edge("web_search_query_generator", "article_chef")

        return workflow.compile()

    async def ainvoke(self, *args, **kwargs):
        return await self.workflow.ainvoke(*args, **kwargs)
