from typing import Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .text_writer import create_text_writer_agent
from .web_search import create_web_search_agent


class ArticlePostabilityGrader(BaseModel):
    """Binary scores for verifying if an article mentions sport name, team names, tournament name, and meets the minimum word count of 100 words."""

    off_or_ontopic: str = Field(
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
    article: str


class OutputFinalArticleState(TypedDict):
    final_article: str
    off_or_ontopic: str


class SharedArticleState(InputArticleState, OutputFinalArticleState):
    mentions_sport_name: str
    mentions_team_names: str
    mentions_tournament_name: str
    web_search_complete: bool
    meets_100_words: str


class NewsWorkflow:
    def __init__(self, llm_model="gpt-4o-mini", temperature=0):
        self.web_search_agent = create_web_search_agent()
        self.text_writer_agent = create_text_writer_agent()
        self.llm_postability = ChatOpenAI(model=llm_model, temperature=temperature)
        self.workflow = self._create_workflow()

    def _create_postability_grader(self):
        prompt_template = """
        You are a grader assessing whether a news article meets the following criteria:
        1. The article is about sports or not. If yes answer, answer with 'yes' for off_or_ontopic, otherwise with 'no'.
        2. The articel explicitly mentions the name of the sport (e.g., "Cricket" or "Football" or "Hockey".etc.). If it is present answer with 'yes' for sport_name_mentioned, otherwise respond with 'no'.
        3. The article explicitly mentions the sport event details, for example, by stating "2 team names" for a team sport or "2 player names" for a individual sport (e.g., "India vs Pakistan" or "Roger Federer vs Rafael Nadal"). If this is present, respond with 'yes' for teams_mentioned; otherwise, respond 'no'.
        4. The article mentions the tournament name (e.g. "ICC Champions Trophy" or "World Cup" or "English Premier League" or "Asia Cup" or "Wimbledon" or "French Open"). If it is present answer with 'yes' for tournament_name_mentioned; otherwise answer with 'no'.
        5. The article contains at least 100 words. If this is met, respond with 'yes' for meets_100_words; otherwise, respond 'no'.

        Provide four binary scores ('yes' or 'no') as follows:
        - off_or_ontopic: 'yes' or 'no' depending on whether the article is related to a sport event.
        - sport_name_mentioned: 'yes' or 'no' depending on whether the article mentions the name of the sport.
        - teams_mentioned: 'yes' or 'no' depending on whether the article mentions the name of 2 players or 2 teams.
        - tournament_name_mentioned: 'yes' or 'no' depending on whether the article mentions the name of the tournament.
        - meets_100_words: 'yes' or 'no' depending on whether the article has at least 100 words.
        """
        postability_system = ChatPromptTemplate.from_messages(
            [("system", prompt_template), ("human", "News Article:\n\n{article}")]
        )
        return postability_system | self.llm_postability.with_structured_output(
            ArticlePostabilityGrader
        )

    async def update_article_state(self, state: SharedArticleState) -> SharedArticleState:
        news_chef = self._create_postability_grader()
        response = await news_chef.ainvoke({"article": state["article"]})
        state["off_or_ontopic"] = response.off_or_ontopic
        state["mentions_sport_name"] = response.sport_name_mentioned
        state["mentions_team_names"] = response.teams_mentioned
        state["mentions_tournament_name"] = response.tournament_name_mentioned
        state["meets_100_words"] = response.meets_100_words
        state["web_search_complete"] = False
        return state

    async def web_search_node(self, state: SharedArticleState) -> SharedArticleState:
        response = await self.web_search_agent.ainvoke({"article": state["article"]})
        state["web_search_complete"] = True
        state["article"] += f"{response['agent_output']}"
        return state

    async def word_count_rewriter_node(self, state: SharedArticleState) -> SharedArticleState:
        response = await self.text_writer_agent.ainvoke({"article": state["article"]})
        state["article"] += f" {response['agent_output']}"
        state["final_article"] = response["agent_output"]
        return state

    def news_chef_decider(self,state: SharedArticleState,) -> Literal["web_searcher", "word_count_rewriter", END]: # type: ignore
        if state["off_or_ontopic"] == "no":
            return END
        elif (
            state["mentions_sport_name"] == "yes" 
            and state["mentions_team_names"] == "yes" 
            and state["mentions_tournament_name"] == "yes"
            ):
            next_node = "web_searcher"
        elif (
            state["meets_100_words"] == "no"
            and state["web_search_complete"] == True
        ):
            next_node = "word_count_rewriter"
        else:
            next_node = END
        return next_node

    def _create_workflow(self):
        workflow = StateGraph(
            SharedArticleState, input=InputArticleState, output=OutputFinalArticleState
        )
        workflow.add_node("news_chef", self.update_article_state)
        workflow.add_node("web_searcher", self.web_search_node)
        workflow.add_node("word_count_rewriter", self.word_count_rewriter_node)
        workflow.set_entry_point("news_chef")
        workflow.add_conditional_edges(
            "news_chef",
            self.news_chef_decider,
            {
                "web_searcher": "web_searcher",
                "word_count_rewriter": "word_count_rewriter",
                END: END,
            },
        )
        workflow.add_edge("web_searcher", "news_chef")
        workflow.add_edge("word_count_rewriter", "news_chef")

        return workflow.compile()

    async def ainvoke(self, *args, **kwargs):
        return await self.workflow.ainvoke(*args, **kwargs)
