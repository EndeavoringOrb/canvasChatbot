# canvasChatbot

export.py will download everything from a course.

prepareDocuments.py will convert each downloaded document into an embedding for RAG.

chat.py allows you to chat with a model that has access to these downloaded documents.

The document retrieval is the main thing that needs to be improved, it does not always retrieve the correct documents.

bot.py is just helper functions for sending queries to the model.