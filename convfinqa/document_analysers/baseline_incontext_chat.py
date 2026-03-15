from abc import ABCMeta, abstractmethod
from typing import List

from data_loaders.convfinqa_original_loader import ParsedItem
from document_analysers.abstract_history_chat import HistoryBasedChat
from llm_client import Response

system_prompt_default = """You are a financial analyst. You will be given a financial report that consists of text as 
well as possibly a table formatted as json that contains further financial data. You will also be given a history of a 
series of previous user questions with your answers and a final user question. Your task is to answer the final user 
question based on the financial report as well as potentially also using the conversation history. The answer of some 
questions could depend on previous answers. Here are some additional rules:

RULES:
- When you encounter thousands, millions or billions in the text don't expand the numbers but return them as mentioned 
in the text but don't copy the thousand, million or billion qualifier. For example, if text says '6.5 billion' then just
return '6.5' without the 'billion'.
- If the number after the decimal is 0 then don't return a float. For example, if '123.0' then just return '123'
- When, the question states something like 'divided' then it expects you to return a percentage 'xx%'.
- Obey the order by which past questions and answers occurred. For example, if you need to subtract two number always 
put first the number that appeared first. 
- If you think that years or dates are not specified clearly enough in the question, then take the earliest and the 
latest year to compute changes or differences.
- When performing calculations you MUST maintain the order by which values appears in the history. For example, if the
question says "what is the differences between the two", then when doing val1-val2, val1 must be mentioned before val2.
- Always return a number. Even if the question asks for percentage do not return something like 14.1%. Return 0.141
- Do not round numbers. Go up to 5 precision points.
- Do not return net differences. Also return negative numbers.
- If the question is a yes/no question then return 1.0 or 0.0.

Here is the financial report:
Financial Report: {report}

Try to think step by step. First, check if the answer is a value from the report in the text or in the table. If not, 
then check if the answer can be computed using some value from the text and/or also some previous answer. 
"""

user_prompt = """User final question: {question}. Be extremely concise and only respond with the answer to the question. 
Don't explain anything and don't use text in your answer. Also return a reason for your answer.
"""


class BaselineInContextChat(HistoryBasedChat, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        doc_as_string = self.format_document(self.document)
        self.system_prompt_compiled = system_prompt_default.format(report=doc_as_string)

    def format_document(self, data_record: ParsedItem) -> str:
        return (
            data_record.pre_text + '\n' + f'<table>\n{str(data_record.table_json)}\n</table>\n' + data_record.post_text
        )

    @abstractmethod
    def build_messages(self, message) -> List:
        pass

    def run_single_turn(self, message) -> Response:
        """Uses LLM to answer a single question based on past history and then records answer to history"""
        messages = self.build_messages(message)
        response = self.llm_inference.answer_question(messages)
        self.update_history(message, response.answer)
        return response


@HistoryBasedChat.register('openai_history_style')
class OpenAIStyleHistoryChat(BaselineInContextChat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_messages(self, message):
        return (
            [{'role': 'system', 'content': self.system_prompt_compiled}]
            + self.history
            + [
                {
                    'role': 'user',
                    'content': user_prompt.format(question=message),
                }
            ]
        )


@HistoryBasedChat.register('embedded_history_style')
class EmbeddedHistoryChat(BaselineInContextChat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history_addition = """Here is the conversation history so far:
        <history>
        {history}
        </history>
        """

    def build_messages(self, message):
        history_str = '\n'.join([f'{msg["role"]}: {msg["content"]}' for msg in self.history])
        history_compiled = self.history_addition.format(history=history_str)
        return [
            {
                'role': 'system',
                'content': self.system_prompt_compiled + history_compiled,
            }
        ] + [{'role': 'user', 'content': user_prompt.format(question=message)}]
