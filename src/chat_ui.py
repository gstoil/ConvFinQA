import os

import dotenv
import gradio as gr

from chat_with_history import HistoryBasedChat
from data_loaders.convfinqa_original_loader import ConvFinQaOriginalLoader

dotenv.load_dotenv('.env')


financial_data_file = os.environ.get('FINANCIAL_DATA_FILE', ConvFinQaOriginalLoader.train_file)


data_loader = ConvFinQaOriginalLoader(financial_data_file)
all_document_ids = [doc.id for doc in data_loader.financial_dataset]
MODEL = 'gpt-4.1-mini'

chat_instances_cache = dict()


def chat_with_history(
    question,
    history,
    doc_id,
    model=MODEL,
    history_strategy='embedded_history_style',
):
    retrieved_doc = data_loader.find_document(doc_id)
    doc_as_string = data_loader.format_document(retrieved_doc)

    # Cache chat instances already generated so that they can be reused.
    chat_instances_cache[doc_id] = chat_instances_cache.get(
        doc_id,
        HistoryBasedChat.registry[history_strategy](document_as_string=doc_as_string, model=model),
    )
    response = chat_instances_cache[doc_id].run_single_turn(question)
    return str(response.answer)


with gr.Blocks() as demo:

    document_id = gr.Dropdown(choices=all_document_ids, label='Select document')

    # Chat Interface
    gr.ChatInterface(fn=chat_with_history, additional_inputs=document_id)

demo.launch()
