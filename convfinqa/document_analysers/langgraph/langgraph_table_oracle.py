from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import Field

from convfinqa.document_analysers.abstract_history_chat import HistoryBasedChat
from convfinqa.document_analysers.langgraph.langgraph_chat import LangGraphChatter, FinancialState
from convfinqa.llm_client import Response


class ExtendedResponse(Response):
    check_table: bool = Field(
        description='Returns True if the LLM cannot find the answer and we need to ask the table agent.'
    )


@HistoryBasedChat.register('langgraph_table_oracle')
class LangGraphTableOracleChat(LangGraphChatter):
    """
    Uses a main agent to check if it can answer question from text otherwise
    it can call a table agent to return answer based on the table.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        table_llm = ChatOpenAI(model=self.model)
        self.table_llm_with_output = table_llm.with_structured_output(Response)

        text_llm = ChatOpenAI(model=self.model)
        self.text_llm_with_output = text_llm.with_structured_output(ExtendedResponse)

    def build_graph(self):
        graph = StateGraph(FinancialState)
        graph.add_node('table_agent', self.table_agent)
        graph.add_node('text_agent', self.text_agent)

        # Schema where text_agent attempts to answer question and if they cannot they invoke the table_agent which then
        # returns the control back to the text_agent
        graph.add_edge(START, 'text_agent')

        graph.add_conditional_edges('text_agent', self.route_after_text, {'table_agent': 'table_agent', 'end': END})
        # important: go back to text_agent
        graph.add_edge('table_agent', 'text_agent')

        return graph

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

    def route_after_text(self, state: FinancialState):

        if state['route'] == 'table':
            return 'table_agent'

        return 'end'

    def text_agent(self, state: FinancialState):
        question = state['question']
        history = state.get('history', [])
        table_answer = state.get('table_answer')
        prompt = """You are analysing a financial report. You will receive data in the form of a text document, a 
        history of user questions and answers and a new user question and your task is to use the history and the 
        text to answer the user question. """
        if table_answer is not None:
            prompt += f"""You will also receive a tentative answer to the question by some other agent that looked the
            answer over some table. The agent will provide its answer along with a reason why it thinks this is the 
            correct answer. Combine the agent's answer with the text and history and try to answer the user question.

            Here is the financial document:
            {self.document.pre_text}
            {self.document.post_text}

            Conversation history:
            {history}

            Table answer:
            {table_answer['answer']} {table_answer['reason']}

            Question:
            {question}

            Answer using the financial table and exploit the history so far to disambiguate if there are co-references.
            """

            response = self.text_llm_with_output.invoke(prompt)

            return {'final_answer': response.model_dump(), 'route': 'done'}

        prompt += f""" Return the answer to user question as well as a reason for your answer. Do not try to guess or
        induce the answer and be cautious. If you think the answer cannot be computed from the text or history, 
        respond with CHECK_TABLE in order to ask some table agent to check and then you will try again later.
        Here is the financial document:
        {self.document.pre_text}
        {self.document.post_text}

        Conversation history:
        {history}

        Question:
        {question}

        Answer using the financial report and exploit the history so far to disambiguate if there are co-references.            
        """
        response = self.text_llm_with_output.invoke(prompt)

        if response.check_table:
            return {'route': 'table'}

        return {'final_answer': response.model_dump(), 'route': 'done'}

    def run_single_turn(self, message) -> Response:
        config = {'configurable': {'thread_id': self.document.id}}
        result = self.compiled_graph.invoke({'question': message}, config=config)
        return Response(**result['final_answer'])
