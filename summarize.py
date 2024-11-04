import requests
from pypdf import PdfReader


API_URL = "https://arc-ai-rag-01.wpi.edu/api/v1/prediction/d1f72c8e-a6af-4db8-ac97-d55e33bc6648"


"""
Returns
text - the response text
question - the question text
chatId
chatMessageId
sessionId
memoryType
"""
def query(payload):

    response = requests.post(API_URL, json=payload)

    return response.json()


def summarizeFile(filePath: str):
    # Get text from file
    print(f"Extracting text from {filePath}")
    if filePath.endswith(".txt"):
        with open(filePath, "r", encoding="utf-8") as f:
            text = f.read()
    elif filePath.endswith(".pdf"):
        reader = PdfReader(filePath)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    else:
        print(f"Can only handle files with the following extensions:")
        print(f"  .txt .pdf")
        return None

    print(f"Extracted {len(text):,} characters")

    print(text)

    # Summarize text using API
    print(f"Getting summary of text")
    summary = query(
        {
            "question": text,
        }
    )
    """
    summary["text"]
    summary["question"]
    """

    return summary["text"]


if __name__ == "__main__":
    summary = summarizeFile(
        "canvasDownloads\\0\\B24\\CS2102-B24\\modules\\Week 1 - 21 Oct - 25 Oct\\files\\SunLecture01.pdf"
    )
    print(summary)
