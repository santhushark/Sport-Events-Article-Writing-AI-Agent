from typing import TypedDict

from langgraph.graph import END, StateGraph

from .article_chef_workflow import ArticleWorkflow


class InputState(TypedDict):
    event: str


class IntermediateState(InputState):
    final_article: str
    error: bool
    ontopic: str


class FinalState(IntermediateState):
    confirmed: str

#Human workflow agent
class HumanWorkflow:
    def __init__(self):
        self.app = ArticleWorkflow()
        self.checkpointer = None
        self.workflow = None

    def set_checkpointer(self, checkpointer):
        self.checkpointer = checkpointer
        self.workflow = self.init_create_workflow()
    
    def init_create_workflow(self):
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        workflow = StateGraph(FinalState, input=InputState, output=FinalState)
        workflow.add_node("newsagent_node", self.newsagent_node)
        workflow.add_node("confirm_node", self.confirm_node)
        workflow.set_entry_point("newsagent_node")
        workflow.add_edge("newsagent_node", "confirm_node")
        workflow.add_edge("confirm_node", END)
        return workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_after=["newsagent_node"],
        )

    async def newsagent_node(self, state: IntermediateState) -> IntermediateState:
        try:
            print("Event: " + state["event"])
            response = await self.app.ainvoke({"event": state["event"]})
            state["final_article"] = response.get(
                "final_article", "Article not relevant for news agency"
            )
            state["ontopic"] = response["ontopic"]
            state["error"] = False
        except Exception as e:
            state["final_Article"] = "Error occured while creating a message"
            state["error"] = True
            print(f"Error invoking newsagent_node: {e}")
        return state

    def confirm_node(self, state: FinalState) -> FinalState:
        state["confirmed"] = "true"
        return state

    async def ainvoke(self, *args, **kwargs):
        if not self.workflow:
            raise RuntimeError("HumanWorkflow has no checkpointer set.")
        return await self.workflow.ainvoke(*args, **kwargs)
