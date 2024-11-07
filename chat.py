from sentence_transformers import SentenceTransformer
from prepareDocuments import getDocuments
import numpy as np
from bot import *


def getSearchQuery(
    dialogue,
):
    query = queryBot(
        createQuery(
            "Respond with a plain text search query that will find documents in the current subject relevant to the user's last entry.",
            dialogue,
            [],
            'plain text search query: "',
            local=True,
        )
    ).split('"')[0]
    return query


def getChatText(dialogue):
    text = ""
    for item in dialogue:
        text += f"{item['name']}: {item['text']}\n"
    return text


class Documents:
    def __init__(self, courseFolder: str) -> None:
        self.courseFolder = courseFolder
        self.courseName = courseFolder.strip("\\/").split("\\")[-1].split("/")[-1]
        self.documents = getDocuments(courseFolder)
        self.documentEmbeddings = np.load(f"embeddings/{self.courseName}.npy")
        self.model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
        )

    def findDocuments(self, query, topN):
        # Get query embedding
        sentences = [f"search_query: {query}"]
        queryEmbedding = self.model.encode(sentences)

        # Get most similar documents
        similarities = (self.documentEmbeddings @ queryEmbedding.T).flatten()
        topNIndices = np.argpartition(similarities, -topN)[-topN:]

        documents = []
        for idx in topNIndices:
            documents.append(self.documents[idx])

        return documents


if __name__ == "__main__":
    print(f"Loading documents")
    documentSearcher = Documents(input("Please enter the course folder path: "))
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
        print(f'searching documents with query "{searchQuery}"')
        documents = documentSearcher.findDocuments(searchQuery, 2)
        for i, doc in enumerate(documents):
            print(f"DOC {i}:")
            print(doc)

        # Get bot response
        print("generating response")
        response = queryBot(
            createQuery(
                "Answer the user. If the DOCUMENTS are relevant, base your response on them.",
                dialogue,
                documents,
                "",
                local,
            )
        ).strip()
        dialogue.append({"name": "assistant", "text": response})
        print(f"Bot: {response}")

        with open("chatOutput.txt", "w", encoding="utf-8") as f:
            f.write(getChatText(dialogue))
