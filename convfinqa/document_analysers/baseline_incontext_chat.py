from abc import ABCMeta

from data_loaders.convfinqa_original_loader import ParsedItem
from document_analysers.abstract_history_chat import HistoryBasedChat
from prompts import system_prompt_default, user_prompt


class BaselineInContextChat(HistoryBasedChat, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        doc_as_string = self.format_document(self.document)
        self.system_prompt_compiled = system_prompt_default.format(report=doc_as_string)

    def format_document(self, data_record: ParsedItem) -> str:
        return (
            data_record.pre_text
            + '\n'
            + f'<table>\n{str(self.table_to_json(data_record.table_ori))}\n</table>\n'
            + data_record.post_text
        )

    @staticmethod
    def table_to_json(table: list[list[str]]) -> dict:
        if len(table) < 2:
            return {}

        headers = [h.strip() for h in table[0][1:]]

        def parse_number(value: str) -> float:
            value = value.strip()
            if value.startswith('(') and value.endswith(')'):
                value = '-' + value[1:-1]
            value = value.replace('$', '').replace(',', '')
            return float(value)

        result = {header: {} for header in headers}

        for row in table[1:]:
            metric = row[0].strip().lower()
            for i, header in enumerate(headers):
                if i + 1 < len(row):
                    try:
                        result[header][metric] = parse_number(row[i + 1])
                    except Exception:
                        continue

        return result


@HistoryBasedChat.register('openai_history_style')
class OpenAIStyleHistoryChat(BaselineInContextChat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.doc

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
