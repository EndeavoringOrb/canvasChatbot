from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from tqdm import tqdm
import numpy as np
from bot import *
import json
import os


def listFilesInDir(directory):
    """
    Returns a list of all filenames (including subdirectories) in the given directory.

    Parameters:
    directory (str): The path to the directory to list files from.

    Returns:
    list: A list of all filenames (including subdirectories) in the given directory.
    """
    fileList = []

    for root, dirs, files in os.walk(directory):
        if root == directory:
            continue  # don't include course json file
        for file in files:
            filePath = os.path.join(root, file)
            fileList.append(filePath)

    return fileList


def summarizeText(text):
    summary = queryBot(
        createQuery(
            "Summarize the given text.", [{"name": "user", "text": text}], [], True
        )
    )
    return summary


def getDocuments(courseFolder: str):
    documents = []

    # Read course json file
    courseName = courseFolder.strip("\\/").split("\\")[-1].split("/")[-1]
    with open(f"{courseFolder}/{courseName}.json", "r", encoding="utf-8") as f:
        courseInfo = json.loads(f.read())

    for item in courseInfo["assignments"]:
        documents.append(str(item))
    for item in courseInfo["announcements"]:
        documents.append(str(item))
    for item in courseInfo["pages"]:
        documents.append(str(item))

    # Read all other files
    fileNames: list[str] = listFilesInDir(courseFolder)
    for name in fileNames:
        if name.endswith(".txt"):
            with open(name, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append(text)
        elif name.endswith(".pdf"):
            reader = PdfReader(name)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents.append(text)

    return documents


def summarizeDocuments(documents):
    return [summarizeText(text) for text in tqdm(documents, desc="Summarizing Documents")]


if __name__ == "__main__":
    print(f"Loading embedding model")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True, weights_only=False)
    courseFolder = "canvasDownloads\\0\\B24\\CS2102-B24"

    # Get summaries
    print(f"Getting documents")
    documents = getDocuments(courseFolder)
    summaries = summarizeDocuments(documents)
    print(summaries)

    # Get summary embeddings
    sentences = [f"search_document: {summary}" for summary in summaries]
    embeddings = model.encode(sentences)
    print(embeddings.shape)

    # Save embeddings
    courseName = courseFolder.strip("\\/").split("\\")[-1].split("/")[-1]
    os.makedirs("embeddings", exist_ok=True)
    np.save(f"embeddings/{courseName}.npy", embeddings)
