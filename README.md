# ConvFinQA Assignment

## Description

Mini project to load and chat with the [ConvFinQA](https://github.com/czyssrs/ConvFinQA) data using modern LLMs.
Currently, implements fairly baseline prompting (0-shot with some rules to help LLM avoid some obvious bad cases). 

### Setup
1. Clone this repository
2. Use the UV environment manager to install dependencies:
```bash
# set up env
uv sync
```
3. Download the ConvFinQA original dataset and put it under folder `/data`
4. Create a `.env` file under root directory and put your openai key
`OPENAI_API_KEY=your_api_key`

### UI

You can start a gradio based UI where you can select the document id over which you want to ask questions.
```bash
 uv run python src/chat_ui.py
```

### UI

You can run automatic evaluation over a sample of the data (default sample size is 50).
The script will compute a few different scores including `ROUGE`