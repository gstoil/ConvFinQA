from unittest.mock import patch

import pytest

from chat_with_history import (
    EmbeddedHistoryChat,
    HistoryBasedChat,
    OpenAIStyleHistoryChat,
)
from llm_client import Response
from prompts import system_prompt, user_prompt

DOC = 'Revenue was $100M in 2023.'
MODEL = 'gpt-4o'


@pytest.fixture
def openai_chat():
    with patch('chat_with_history.LLMInference'):
        return OpenAIStyleHistoryChat(DOC, MODEL)


@pytest.fixture
def embedded_chat():
    with patch('chat_with_history.LLMInference'):
        return EmbeddedHistoryChat(document_as_string=DOC, model=MODEL)


class TestRegistry:
    def test_openai_style_registered(self):
        assert 'openai_history_style' in HistoryBasedChat.registry
        assert HistoryBasedChat.registry['openai_history_style'] is OpenAIStyleHistoryChat

    def test_embedded_style_registered(self):
        assert 'embedded_history_style' in HistoryBasedChat.registry
        assert HistoryBasedChat.registry['embedded_history_style'] is EmbeddedHistoryChat


class TestUpdateHistory:
    def test_appends_user_and_assistant(self, openai_chat):
        openai_chat.update_history('What was revenue?', '100M')
        assert openai_chat.history == [
            {'role': 'user', 'content': 'What was revenue?'},
            {'role': 'assistant', 'content': '100M'},
        ]

    def test_multiple_turns_accumulate(self, openai_chat):
        openai_chat.update_history('Q1', 'A1')
        openai_chat.update_history('Q2', 'A2')
        assert len(openai_chat.history) == 4

    def test_history_starts_empty(self, openai_chat):
        assert openai_chat.history == []


class TestOpenAIStyleHistoryChat:
    def test_build_messages_empty_history_length(self, openai_chat):
        messages = openai_chat.build_messages('What is revenue?')
        assert len(messages) == 2

    def test_build_messages_roles(self, openai_chat):
        messages = openai_chat.build_messages('What is revenue?')
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'

    def test_system_prompt_contains_document(self, openai_chat):
        messages = openai_chat.build_messages('Q')
        assert DOC in messages[0]['content']

    def test_system_prompt_compiled_correctly(self, openai_chat):
        messages = openai_chat.build_messages('Q')
        assert messages[0]['content'] == system_prompt.format(report=DOC)

    def test_user_prompt_contains_question(self, openai_chat):
        question = 'What is revenue?'
        messages = openai_chat.build_messages(question)
        assert question in messages[-1]['content']

    def test_user_prompt_compiled_correctly(self, openai_chat):
        question = 'What is revenue?'
        messages = openai_chat.build_messages(question)
        assert messages[-1]['content'] == user_prompt.format(question=question)

    def test_build_messages_with_history_length(self, openai_chat):
        openai_chat.history = [
            {'role': 'user', 'content': 'Q1'},
            {'role': 'assistant', 'content': 'A1'},
        ]
        messages = openai_chat.build_messages('Q2')
        assert len(messages) == 4  # system + 2 history + user

    def test_history_inserted_between_system_and_user(self, openai_chat):
        openai_chat.history = [
            {'role': 'user', 'content': 'Q1'},
            {'role': 'assistant', 'content': 'A1'},
        ]
        messages = openai_chat.build_messages('Q2')
        assert messages[1] == {'role': 'user', 'content': 'Q1'}
        assert messages[2] == {'role': 'assistant', 'content': 'A1'}
        assert messages[3]['role'] == 'user'

    def test_history_order_preserved(self, openai_chat):
        openai_chat.history = [
            {'role': 'user', 'content': 'Q1'},
            {'role': 'assistant', 'content': 'A1'},
            {'role': 'user', 'content': 'Q2'},
            {'role': 'assistant', 'content': 'A2'},
        ]
        messages = openai_chat.build_messages('Q3')
        assert messages[1]['content'] == 'Q1'
        assert messages[2]['content'] == 'A1'
        assert messages[3]['content'] == 'Q2'
        assert messages[4]['content'] == 'A2'


class TestEmbeddedHistoryChat:
    def test_build_messages_empty_history_length(self, embedded_chat):
        messages = embedded_chat.build_messages('What is revenue?')
        assert len(messages) == 2

    def test_build_messages_roles(self, embedded_chat):
        messages = embedded_chat.build_messages('Q')
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'

    def test_system_prompt_contains_document(self, embedded_chat):
        messages = embedded_chat.build_messages('Q')
        assert DOC in messages[0]['content']

    def test_user_prompt_contains_question(self, embedded_chat):
        question = 'What is revenue?'
        messages = embedded_chat.build_messages(question)
        assert question in messages[-1]['content']

    def test_history_embedded_in_system_prompt(self, embedded_chat):
        embedded_chat.history = [
            {'role': 'user', 'content': 'Q1'},
            {'role': 'assistant', 'content': 'A1'},
        ]
        messages = embedded_chat.build_messages('Q2')
        assert 'Q1' in messages[0]['content']
        assert 'A1' in messages[0]['content']

    def test_history_not_separate_messages(self, embedded_chat):
        embedded_chat.history = [
            {'role': 'user', 'content': 'past question'},
            {'role': 'assistant', 'content': 'past answer'},
        ]
        messages = embedded_chat.build_messages('new question')
        # History is embedded in system prompt, not added as extra messages
        assert len(messages) == 2

    def test_history_roles_included_in_system_prompt(self, embedded_chat):
        embedded_chat.history = [
            {'role': 'user', 'content': 'Q1'},
            {'role': 'assistant', 'content': 'A1'},
        ]
        messages = embedded_chat.build_messages('Q2')
        assert 'user' in messages[0]['content']
        assert 'assistant' in messages[0]['content']

    def test_empty_history_no_stale_content(self, embedded_chat):
        messages = embedded_chat.build_messages('Q')
        # With no history, the history block should be empty
        system_content = messages[0]['content']
        assert 'user:' not in system_content
        assert 'assistant:' not in system_content


class TestRunSingleTurn:
    def test_returns_response(self, openai_chat):
        mock_response = Response(answer='100M', reason='from report')
        openai_chat.llm_inference.answer_question.return_value = mock_response
        result = openai_chat.run_single_turn('What is revenue?')
        assert result is mock_response

    def test_history_updated_after_turn(self, openai_chat):
        mock_response = Response(answer='100M', reason='from report')
        openai_chat.llm_inference.answer_question.return_value = mock_response
        openai_chat.run_single_turn('What is revenue?')
        assert openai_chat.history[-2] == {
            'role': 'user',
            'content': 'What is revenue?',
        }
        assert openai_chat.history[-1] == {
            'role': 'assistant',
            'content': '100M',
        }

    def test_answer_question_called_with_messages(self, openai_chat):
        mock_response = Response(answer='100M', reason='from report')
        openai_chat.llm_inference.answer_question.return_value = mock_response
        openai_chat.run_single_turn('What is revenue?')
        openai_chat.llm_inference.answer_question.assert_called_once()
        call_args = openai_chat.llm_inference.answer_question.call_args[0][0]
        assert isinstance(call_args, list)
        assert call_args[-1]['role'] == 'user'

    def test_multi_turn_history_grows(self, openai_chat):
        openai_chat.llm_inference.answer_question.return_value = Response(answer='A1', reason='')
        openai_chat.run_single_turn('Q1')
        openai_chat.llm_inference.answer_question.return_value = Response(answer='A2', reason='')
        openai_chat.run_single_turn('Q2')
        assert len(openai_chat.history) == 4

    def test_second_turn_includes_first_in_messages(self, openai_chat):
        openai_chat.llm_inference.answer_question.return_value = Response(answer='A1', reason='')
        openai_chat.run_single_turn('Q1')
        openai_chat.llm_inference.answer_question.return_value = Response(answer='A2', reason='')
        openai_chat.run_single_turn('Q2')
        call_args = openai_chat.llm_inference.answer_question.call_args[0][0]
        contents = [m['content'] for m in call_args]
        assert any('Q1' in c for c in contents)
