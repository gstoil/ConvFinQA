import dotenv
import gradio as gr

from chat_with_history import HistoryBasedChat
from data_loaders.convfinqa_original_loader import ConvFinQaOriginalLoader

dotenv.load_dotenv('.env')


data_loader = ConvFinQaOriginalLoader()
all_document_ids = [doc.id for doc in data_loader.financial_dataset]
MODEL = 'gpt-4.1-mini'

chat_instances_docs_map = dict()


def chat_with_history(
    question,
    history,
    doc_id,
    model=MODEL,
    history_strategy='embedded_history_style',
):
    retrieved_doc = data_loader.find_document(doc_id)
    doc_as_string = data_loader.format_document(retrieved_doc)

    # store the chat instances already generated for various docs.
    chat_instances_docs_map[doc_id] = chat_instances_docs_map.get(
        doc_id,
        HistoryBasedChat.registry[history_strategy](document_as_string=doc_as_string, model=model),
    )
    response = chat_instances_docs_map[doc_id].run_single_turn(question)
    return response.answer


with gr.Blocks() as demo:

    document_id = gr.Dropdown(choices=all_document_ids, label='Select document')

    # Chat Interface
    gr.ChatInterface(fn=chat_with_history, additional_inputs=document_id)

demo.launch()
