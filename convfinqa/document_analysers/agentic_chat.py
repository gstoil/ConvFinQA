from typing import TypedDict, Optional, List
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from document_analysers.abstract_history_chat import HistoryBasedChat
from llm_client import Response


class QAItem(TypedDict):
    question: str
    answer: str
    reason: str


class FinancialState(TypedDict):
    question: str

    table_answer: Optional[dict]
    text_answer: Optional[dict]
    final_answer: Optional[dict]

    history: List[QAItem]


@HistoryBasedChat.register('langgraph_chat')
class LangGraphChat(HistoryBasedChat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        graph = StateGraph(FinancialState)

        # Step 3
        table_llm = ChatOpenAI(model=self.model)
        self.table_llm_with_output = table_llm.with_structured_output(Response)

        text_llm = ChatOpenAI(model=self.model)
        self.text_llm_with_output = text_llm.with_structured_output(Response)

        aggregator_llm = ChatOpenAI(model=self.model)
        self.aggregator_llm_with_output = aggregator_llm.with_structured_output(Response)

        graph.add_node('table_agent', self.table_agent)
        graph.add_node('text_agent', self.text_agent)
        graph.add_node('aggregator', self.aggregator)

        graph.add_edge(START, 'table_agent')
        graph.add_edge(START, 'text_agent')

        graph.add_edge('table_agent', 'aggregator')
        graph.add_edge('text_agent', 'aggregator')

        # Step 5 add memory
        memory = MemorySaver()
        self.app = graph.compile(checkpointer=memory)

    def table_agent(self, state: FinancialState):
        question = state['question']
        history = state.get('history', [])

        prompt = f"""
        You are a financial analyst specialising in answering questions using financial tables. You will receive data
        in the form of a Json-table, a history of user questions and answers and a user question and your task is to 
        use the history and user question and decide if the user question refers to the provided table. If so then you
        should return the answer found in the table. The table will have the following format:
        {{row_1: {{column_1: value, column_2: value, ...}},
          row_2: {{column_1: value, column_2: value, ...}}
          ...
        }}

            Here is the table:
            {self.document.table_json}

            Here is the conversation so far:
            {history}

            Here is the user question:
            {question}
            Answer using the financial table and exploit the history so far to disambiguate if there are co-references.
        """

        response = self.table_llm_with_output.invoke(prompt)

        return {'table_answer': response.model_dump()}

    def text_agent(self, state: FinancialState):
        question = state['question']
        history = state.get('history', [])

        prompt = f"""You are analysing a financial report. You will receive data in the form of a text document, a 
        history of user questions and answers and a user question and your task is to use the history the question 
        and the text to answer the user question. Return the answer to user question as well as a reason for your 
        answer.

        Here is the financial document:

        {self.document.pre_text}
        {self.document.post_text}

        Conversation history:
        {history}

        Question:
        {question}

        Answer using the financial table and exploit the history so far to disambiguate if there are co-references.
        """

        response = self.text_llm_with_output.invoke(prompt)

        return {'text_answer': response.model_dump()}

    def aggregator(self, state: FinancialState):
        question = state['question']
        table_answer = state['table_answer']
        text_answer = state['text_answer']
        history = state.get('history', [])

        prompt = f"""Question:
        {question}

        Answer from table agent and its reasoning:
        {table_answer['answer']} {table_answer['reason']}

        Answer from text agent:
        {text_answer['answer']} {text_answer['reason']}

        Choose the best answer or combine them.
        """

        response = self.aggregator_llm_with_output.invoke(prompt)

        response_dict = response.model_dump()
        new_history = history + [
            {'question': question, 'answer': response_dict['answer'], 'reason': response_dict['reason']}
        ]

        return {'final_answer': response.model_dump(), 'history': new_history}

    def run_single_turn(self, message) -> Response:
        config = {'configurable': {'thread_id': self.document.id}}
        result = self.app.invoke({'question': message}, config=config)
        return Response(**result['final_answer'])
