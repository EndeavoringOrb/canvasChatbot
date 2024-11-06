from llama_cpp import Llama
from datetime import date
import requests

API_URL = "https://arc-ai-rag-01.wpi.edu/api/v1/prediction/d1f72c8e-a6af-4db8-ac97-d55e33bc6648"

llm: Llama = Llama(
    model_path="C:/Users/aaron/CODING/llms/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    n_ctx=8192,
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
    documents = f"\n\nDOCUMENTS:\n{documents}" if len(query["documents"]) > 0 else ""
    text += f"<|start_header_id|>{query['dialogue'][-1]['name']}<|end_header_id|>\n\n{query['dialogue'][-1]['text']}{documents}<|eot_id|>"
    text += "<|start_header_id|>assistant<|end_header_id|>"
    return text


def queryBot(query):
    if query["local"]:
        text = formatPromptLlama3_2(query)
        print("\n" * 10)
        print(text)
        print("\n" * 10)
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
        text += "Bot: "
        return queryAPI({"question": text})["text"]


def createQuery(
    systemPrompt: str, dialogue: list[dict[str, str]], documents: list[str], local: bool
):
    return {
        "systemPrompt": systemPrompt,
        "dialogue": dialogue,
        "documents": documents,
        "local": local,
    }
