from sentence_transformers import SentenceTransformer
from prepareDocuments import getDocuments
import numpy as np
from bot import *


def getSearchQuery(
    dialogue,
):
    query = queryBot(
        createQuery(
            "Respond with a search query that will find documents relevant to the user's last entry.",
            dialogue,
            [],
            local=True,
        )
    )
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
            "nomic-ai/nomic-embed-text-v1", 
            trust_remote_code=True,
            weights_only=False
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
    documentSearcher = Documents("canvasDownloads\\0\\B24\\CS2102-B24")
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
        print(f"searching documents with query \"{searchQuery}\"")
        documents = documentSearcher.findDocuments(searchQuery, 1)

        # Get bot response
        print("generating response")
        response = queryBot(
            createQuery(
                "Answer the user based on the DOCUMENTS.", dialogue, documents, local
            )
        ).strip()
        dialogue.append({"name": "assistant", "text": response})
        print(f"Bot: {response}")

        with open("chatOutput.txt", "w", encoding="utf-8") as f:
            f.write(getChatText(dialogue))
