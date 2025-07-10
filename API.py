# Making an api to be called on the website
# Import modules
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
import os

# define max number of turns to retain
MAX_TURNS = 3

# defines embedding model to make database
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# create a database folder if not existing
os.makedirs("Database", exist_ok=True)

# if vector database exists then load that or else create one
if os.path.exists('Database/MyVectorDB.pkl') and os.path.exists('Database/MyVectorDB.faiss'):
    db = FAISS.load_local("Database", model, "MyVectorDB", allow_dangerous_deserialization=True)
else:
    paths = [
        r".\Documents\Attention_is_all_you_need.pdf",
        r".\Documents\BERT.pdf",
        r".\Documents\Contrastive_Language.pdf",
        r".\Documents\GPT_3.pdf",
        r".\Documents\LLaMa.pdf"
    ]

    documents = []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # load each of the 5 docs and split them into 1000 chunk size with 200 overlap
    for path in paths:
        loader = PyPDFLoader(path)

        documents += loader.load_and_split(text_splitter)

    # create FAISS database
    db = FAISS.from_documents(documents, model)

    # persist database
    db.save_local("./Database", "MyVectorDB")

# initialize llama3
llm = ChatOllama(
    model="llama3",
    temperature=0.4
)

# retain messages with first a system order on how to behave
messages = [
    ("system",
     "You are answering questions on 5 documents of research. You will be given 5 context documents separated by '\n\n' formulate an appropriate reply. These may or may not be related to the prior conversation. So try to stick as closely as possible to the latest asked questions.")
]

# write an ask function
def ask(query):
    global messages

    # retrieve top 5 similar documents and their scores
    retrieved = db.similarity_search_with_score(query, 5)
    context = "\n\n".join([i[0].page_content for i in retrieved])
    prompt = f"Question: {query}\nContext: \n" + context

    # if total number of turns exceeds max number of terms then add a summarising prompt to summarise all past msgs into a single msg
    # to retain context
    if len(messages) >= MAX_TURNS * 2:
        messages.append(("human", "Distill the above chat messages into a single summary message. Include as many specific details as you can."))
        ai_msg = llm.invoke(messages)
        messages = [("ai", ai_msg.content)]
        print("Previous chats trimmed")

    # add query to global messages
    messages.append(("human", query))

    # create copy of global messages and add context only for the latest query from retrieved docs
    prompt_chain = messages + [("human", prompt)]

    ai_msg = llm.invoke(prompt_chain)

    messages.append(("ai", ai_msg.content))

    # returning documents with their scores and the AI message
    docs = [{"title": doc[0].page_content[:100]+"...", "full": doc[0].page_content, "score": doc[1]} for doc in retrieved]

    return [ai_msg.content, docs]
