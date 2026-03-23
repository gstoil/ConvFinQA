from abc import abstractmethod
from typing import TypedDict, Optional, List

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from convfinqa.document_analysers.abstract_history_chat import HistoryBasedChat
from convfinqa.llm_client import Response


class QAItem(TypedDict):
    question: str
    answer: str
    reason: str


class FinancialState(TypedDict):
    question: str

    table_answer: Optional[dict]
    text_answer: Optional[dict]
    final_answer: Optional[dict]

    route: Optional[str]

    history: List[QAItem]


class LangGraphChatter(HistoryBasedChat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        graph = self.build_graph()

        memory = MemorySaver()
        self.compiled_graph = graph.compile(checkpointer=memory)

    @abstractmethod
    def build_graph(self) -> StateGraph:
        pass

    def run_single_turn(self, message) -> Response:
        config = {'configurable': {'thread_id': self.document.id}}
        result = self.compiled_graph.invoke({'question': message}, config=config)
        return Response(**result['final_answer'])
