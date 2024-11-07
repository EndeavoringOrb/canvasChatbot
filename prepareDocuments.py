from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
from pypdf import PdfReader
from tqdm import tqdm
import numpy as np
from bot import *
import html2text
import pptx2txt2
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
            "Summarize the document given by the user. Even if there is no text, include the info above the text.",
            [{"name": "user", "text": ""}],
            [text],
            "",
            True,
        )
    )
    return summary


def getDocuments(courseFolder: str):
    """
    Converts files/assignments/announcements/pages in the specified folder.
    """
    documents = []

    # Read course json file
    courseName = courseFolder.strip("\\/").split("\\")[-1].split("/")[-1]
    with open(f"{courseFolder}/{courseName}.json", "r", encoding="utf-8") as f:
        courseInfo = json.loads(f.read())

    # Get all filenames
    fileNames: list[str] = listFilesInDir(courseFolder)

    totalDocuments = (
        len(courseInfo["assignments"])
        + len(courseInfo["announcements"])
        + len(courseInfo["pages"])
        + len(fileNames)
    )

    with tqdm(total=totalDocuments, desc="  Getting Documents") as pbar:
        for item in courseInfo["assignments"]:
            text = "ASSIGNMENT\n"

            # Title
            text += "Title: "
            text += item["title"] + "\n"

            # Time
            text += "Created Date: "
            text += (
                datetime.strptime(item["assigned_date"], "%B %d, %Y %I:%M %p")
                - timedelta(hours=5)
            ).strftime("%B %d, %Y %I:%M %p") + "\n"

            # Due Date
            text += "Due Date: "
            text += (
                (
                    datetime.strptime(item["due_date"], "%B %d, %Y %I:%M %p")
                    - timedelta(hours=5)
                ).strftime("%B %d, %Y %I:%M %p")
                + "\n"
                if item["due_date"] != ""
                else "None" + "\n"
            )

            # Description
            text += "Description:\n"
            text += html2text.html2text(item["description"]) + "\n"

            # Submissions
            text += "Submissions:\n" if len(item["submissions"]) > 0 else ""
            for i, submission in enumerate(item["submissions"]):
                text += f"Submission {i}:\n"
                if submission["grade"] == "None":
                    text += f"  Grade: Awaiting Grading\n"
                else:
                    text += f"  Grade: {submission['grade']}/{submission['total_possible_points']}\n"
                text += f"  Comments: {submission['submission_comments']}\n"

            documents.append(text.strip())
            pbar.update(1)
        for item in courseInfo["announcements"]:
            text = "ANNOUNCEMENT\n"

            # Title
            text += "Title: "
            text += item["title"] + "\n"

            # Time
            text += "Created Date: "
            text += (
                datetime.strptime(item["posted_date"], "%B %d, %Y %I:%M %p")
                - timedelta(hours=5)
            ).strftime("%B %d, %Y %I:%M %p") + "\n"

            # Author
            text += "Author: "
            text += item["author"] + "\n"

            # Text
            text += "Text:\n"
            text += html2text.html2text(item["body"]) + "\n"

            documents.append(text.strip())
            pbar.update(1)
        for item in courseInfo["pages"]:
            text = "PAGE\n"

            # Title
            text += "Title: "
            text += item["title"] + "\n"

            # Time
            text += "Created Date: "
            text += (
                datetime.strptime(item["created_date"], "%B %d, %Y %I:%M %p")
                - timedelta(hours=5)
            ).strftime("%B %d, %Y %I:%M %p") + "\n"
            text += "Last Updated Date: "
            text += (
                datetime.strptime(item["last_updated_date"], "%B %d, %Y %I:%M %p")
                - timedelta(hours=5)
            ).strftime("%B %d, %Y %I:%M %p") + "\n"

            # Text
            text += "Text:\n"
            text += html2text.html2text(item["body"]) + "\n"

            documents.append(text.strip())
            pbar.update(1)

        # Read all other files
        for name in fileNames:
            text = "FILE:\n"
            if name.endswith(".txt"):
                with open(name, "r", encoding="utf-8") as f:
                    text += f.read()
                documents.append(text)
            elif name.endswith(".pdf"):
                reader = PdfReader(name)
                for page in reader.pages:
                    text += page.extract_text()
                documents.append(text)
            elif name.endswith(".pptx"):
                text += pptx2txt2.extract_text(name)
                documents.append(text)
            else:
                print(f"Could not parse {name}")
            pbar.update(1)

    return documents


def getDocumentTitles(courseFolder: str):
    """
    Converts files/assignments/announcements/pages in the specified folder.
    """
    documents = []

    # Read course json file
    courseName = courseFolder.strip("\\/").split("\\")[-1].split("/")[-1]
    with open(f"{courseFolder}/{courseName}.json", "r", encoding="utf-8") as f:
        courseInfo = json.loads(f.read())

    # Get all filenames
    fileNames: list[str] = listFilesInDir(courseFolder)

    totalDocuments = (
        len(courseInfo["assignments"])
        + len(courseInfo["announcements"])
        + len(courseInfo["pages"])
        + len(fileNames)
    )

    with tqdm(total=totalDocuments, desc="  Getting Documents") as pbar:
        for item in courseInfo["assignments"]:
            documents.append(item["title"].strip())
            pbar.update(1)
        for item in courseInfo["announcements"]:
            documents.append(item["title"].strip())
            pbar.update(1)
        for item in courseInfo["pages"]:
            documents.append(item["title"].strip())
            pbar.update(1)

        # Read all other files
        for name in fileNames:
            text = name.split("\\")[-1].split("/")[-1].split(".")[0]
            if name.endswith(".txt"):
                documents.append(text)
            elif name.endswith(".pdf"):
                documents.append(text)
            elif name.endswith(".pptx"):
                documents.append(text)
            else:
                print(f"Could not parse {name}")
            pbar.update(1)

    return documents


def summarizeDocuments(documents):
    summaries = []
    for text in tqdm(documents, desc="  Summarizing Documents"):
        try:
            summaries.append(summarizeText(text))
            print(summaries[-1])
        except ValueError as e:
            print(f"ERROR:\n{e}\nSTOPPING SUMMARIZATION")
            break
    return summaries


if __name__ == "__main__":
    titles = input("Use only document titles for embedding? [y/n]: ").lower() == "y"
    print(f"Loading embedding model")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    courseFolders = [
        # "canvasDownloads\\2\\Default Term\\SESA-Projects",
        # "canvasDownloads\\1\\Default Term\\wkgp-sesa",
        "canvasDownloads\\0\\B24\\CS2102-B24",
        # "canvasDownloads\\0\\B24\\MA2051-B24-BL02 (group 2)",
    ]

    for courseFolder in courseFolders:
        print(f"Preparing {courseFolder}")

        # Get documents
        if titles:
            documents: list[str] = getDocumentTitles(courseFolder)
        else:
            documents: list[str] = getDocuments(courseFolder)

        print(f"  Got {len(documents):,} documents.")
        documentLengths = [len(llm.tokenize(doc.encode())) for doc in documents]

        print(
            f"  Avg. Document # Tokens: {int(sum(documentLengths) / len(documents)):,}"
        )
        print(f"  Max. Document # Tokens: {max(documentLengths):,}")
        # documents = summarizeDocuments(documents)

        # Get embeddings
        print(f"  Getting Embeddings")
        sentences = [f"search_document: {document}" for document in documents]
        embeddings = model.encode(sentences, batch_size=1, show_progress_bar=True)
        print(embeddings.shape)

        # Save embeddings
        courseName = courseFolder.strip("\\/").split("\\")[-1].split("/")[-1]
        os.makedirs("embeddings", exist_ok=True)
        np.save(f"embeddings/{courseName}{'-titles' if titles else ''}.npy", embeddings)
