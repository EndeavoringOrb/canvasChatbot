from llama_cpp import Llama
from datetime import date
import requests

API_URL = "a secret"

llm: Llama = Llama(
    model_path=input("Please enter path to .gguf file of model: "),
    n_ctx=32768,
    verbose=False,
)


def queryAPI(payload):
    """
    response has
        text - the response text
        question - the question text
        chatId
        chatMessageId
        sessionId
        memoryType
    """

    response = requests.post(API_URL, json=payload)

    return response.json()


def formatPromptLlama3_2(query):
    text = f"""<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: {date.today().strftime("%d %b %Y")}

{query["systemPrompt"]}<|eot_id|>
"""
    for item in query["dialogue"][:-1]:
        text += f"<|start_header_id|>{item['name']}<|end_header_id|>\n\n{item['text']}<|eot_id|>"
    documents = "\n\n".join(query["documents"])
    documents = (
        f"\n\nDOCUMENTS:\n{documents}\nEND DOCUMENTS"
        if len(query["documents"]) > 0
        else ""
    )
    text += f"<|start_header_id|>{query['dialogue'][-1]['name']}<|end_header_id|>\n\n{query['dialogue'][-1]['text']}{documents}<|eot_id|>"
    text += "<|start_header_id|>assistant<|end_header_id|>"
    text += query["responseStart"]
    return text


def queryBot(query):
    if query["local"]:
        text = formatPromptLlama3_2(query)
        output = llm(
            text,  # Prompt
            max_tokens=None,  # Generate up to max_tokens new tokens
            stop=["<|eot_id|>"],
            echo=False,  # Echo the prompt back in the output
            stream=True,
        )
        outText = ""
        for out in output:
            text = out["choices"][0]["text"]
            outText += text
        return outText
    else:
        text = ""
        text += f"{query['systemPrompt']}\n"
        text += "\n\n".join(query["documents"])
        for item in query["dialogue"]:
            text += f"{item['name']}: {item['text']}\n"
        text += f"Bot: {query['responseStart']}"
        return queryAPI({"question": text})["text"]


def createQuery(
    systemPrompt: str,
    dialogue: list[dict[str, str]],
    documents: list[str],
    responseStart: str = "",
    local: bool = True,
):
    return {
        "systemPrompt": systemPrompt,
        "dialogue": dialogue,
        "documents": documents,
        "responseStart": responseStart,
        "local": local,
    }
