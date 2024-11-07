from sentence_transformers import SentenceTransformer
from prepareDocuments import getDocuments
import numpy as np
from bot import *


def getSearchQuery(
    dialogue,
):
    query = queryBot(
        createQuery(
            "Respond with a search query to find documents relevant to the user's last entry.\nExamples:\nWhen is Homework 2 due? -> Search Query: Homework 2 Due Date\nWhat do we have to do for Lab 0? -> Search Query: Lab 0 Assignment Description\nWhat do I have to do to complete Homework 2? -> Search Query: Homework 2 completion instructions",
            dialogue,
            [],
            'Search Query: "',
            local=True,
        )
    )
    completeQuery = ""
    for item in query:
        completeQuery += item
    completeQuery = completeQuery.split('"')[0]
    return completeQuery


def getChatText(dialogue):
    text = ""
    for item in dialogue:
        text += f"{item['name']}: {item['text']}\n"
    return text


class Documents:
    def __init__(self, courseFolder: str) -> None:
        self.courseFolder = courseFolder
        self.courseName = courseFolder.strip("\\/").split("\\")[-1].split("/")[-1]
        print(f"Loading documents")
        self.documents = getDocuments(courseFolder)
        self.documentsLower = [doc.lower() for doc in self.documents]
        print(f"Loading embeddings")
        try:
            self.documentEmbeddings = np.load(f"embeddings/{self.courseName}.npy")
        except FileNotFoundError:
            self.documentEmbeddings = None
        try:
            self.titleEmbeddings = np.load(f"embeddings/{self.courseName}-titles.npy")
        except FileNotFoundError:
            self.titleEmbeddings = None

        print(f"Loading embedding model")
        self.model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
        )
        print(f"Finished preparation\n")

    def findDocuments(self, query: str, topN: int):
        # Get query embedding
        sentences = [f"search_query: {query}"]
        queryEmbedding = self.model.encode(sentences)

        # Get most similar documents

        # Check document similarity to title
        documentSimilarities = (
            (self.documentEmbeddings @ queryEmbedding.T).flatten()
            if self.documentEmbeddings is not None
            else np.zeros(len(self.documents))
        )

        # Check query similarity to title
        titleSimilarities = (
            (self.titleEmbeddings @ queryEmbedding.T).flatten()
            if self.titleEmbeddings is not None
            else np.zeros(len(self.documents))
        )

        # Check if query words are in documents
        queryWords = query.split()
        wordSimilarities = np.zeros(len(self.documents))
        if len(queryWords) > 0:
            for i, doc in enumerate(self.documentsLower):
                for word in queryWords:
                    if word.lower() in doc:
                        wordSimilarities[i] += 1
            wordSimilarities *= 1.0 / len(queryWords)

        documentSimilarities *= 1
        titleSimilarities *= 1.5
        wordSimilarities *= 0.3
        similarities = documentSimilarities + titleSimilarities + wordSimilarities

        topNIndices = np.flip(
            np.argpartition(similarities, -topN)[-topN:]
        )  # Get top N indices ordered from highest score first to lowest score last
        print(
            "indices, scores, docScores, titleScores, wordScores:",
            topNIndices,
            similarities[topNIndices],
            documentSimilarities[topNIndices],
            titleSimilarities[topNIndices],
            wordSimilarities[topNIndices],
        )

        documents = []
        for idx in topNIndices:
            documents.append(self.documents[idx])

        return documents


if __name__ == "__main__":
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
        documents = documentSearcher.findDocuments(searchQuery, 3)
        for i, doc in enumerate(documents):
            print(f"DOC {i}:")
            print(doc)

        # Get bot response
        print(f"generating response")
        response = queryBot(
            createQuery(
                "Respond to the user. If the DOCUMENTS are relevant, base your response on them.",
                dialogue,
                documents,
                "",
                local,
            )
        )
        print(f"Bot: ")
        completeResponse = ""
        for item in response:
            completeResponse += item
            print(item, end="", flush=True)
        print()
        completeResponse = completeResponse.strip()
        dialogue.append({"name": "assistant", "text": completeResponse})

        with open("chatOutput.txt", "w", encoding="utf-8") as f:
            f.write(getChatText(dialogue))
