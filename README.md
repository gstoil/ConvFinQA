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

### Evaluation

You can run automatic evaluation over a sample of the data. The script will compute a few different scores including 
exact value match, `ROUGE`, and levenshtein similarity. 

Evaluation results for the `dev.json` data (421 test points) and using `gpt-4.1-mini`:

| Metric    | Score |
|-----------|-------|
| exact val | 0.531 |
| lev_sim   | 0.760 |
| rouge     | 0.714 |

Lev sim can capture cases where the difference between expected and computed answer lies in small fractions, e.g., 
`1.04852` vs `1.04868` but could also produce false positives if the difference is like `144` vs `844`.
