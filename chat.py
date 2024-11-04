from llama_cpp import Llama
from datetime import date
import requests
import json

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
    text += f"<|start_header_id|>{query['dialogue'][-1]['name']}<|end_header_id|>\n\n{query['dialogue'][-1]['text']}\n\nDOCUMENTS:{documents}<|eot_id|>"
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


def isMatch(item, query, searchField):
    return item[searchField] == query


def search(searchQuery):
    section, searchField, query, returnField = (
        searchQuery["section"],
        searchQuery["searchField"],
        searchQuery["query"],
        searchQuery["returnField"],
    )
    # Read file
    with open(courseJson, "r", encoding="utf-8") as f:
        courseInfo = json.loads(f.read())

    # Get relevant section
    items = courseInfo[section]

    # Search for relevant items in section
    returnData = []
    for item in items:
        if isMatch(item, query, searchField):
            returnData.append(item[returnField])

    # Return relevant items
    return returnData


def getSearchQuery(context):
    return {
        "section": "assignments",
        "searchField": "title",
        "query": "Lab 0 - Setting up a Java Project   Unit Tests",
        "returnField": "description",
    }


def createQuery(systemPrompt, dialogue, documents, local):
    return {
        "systemPrompt": systemPrompt,
        "dialogue": dialogue,
        "documents": documents,
        "local": local,
    }


def getChatText(dialogue):
    text = ""
    for item in dialogue:
        text += f"{item['name']}: {item['text']}\n"
    return text


courseJson = "canvasDownloads\\0\\B24\\CS2102-B24\\CS2102-B24.json"

if __name__ == "__main__":
    local = True
    dialogue = []
    documents = []
    while True:
        # Get user input
        userInput = input("User: ")
        dialogue.append({"name": "user", "text": userInput})

        # Get search query based on context using a specialized bot
        print("getting search query")
        searchQuery = getSearchQuery(dialogue)

        # Search for documents
        print("searching documents")
        documents = search(searchQuery)

        # Get bot response
        print("generating response")
        response = queryBot(
            createQuery(
                "Answer the user based on the DOCUMENTS.", dialogue, documents, local
            )
        )
        dialogue.append({"name": "assistant", "text": response})

        with open("chatOutput.txt", "w", encoding="utf-8") as f:
            f.write(getChatText(dialogue))
