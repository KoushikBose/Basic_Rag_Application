from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()  # load HF_TOKEN / GROQ_API_KEY from .env
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import glob
import pandas as pd



# 🔵 1. Re-initialize Retriever
# 🟢 Load every PDF from the "knowledge base" folder (one level above /backend)
# 🟢 instead of the old /content/langchain_rag_dataset.txt file

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "knowledge base")
raw_doc = []
for pdf_path in glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.pdf")):
    raw_doc.extend(PyPDFLoader(pdf_path).load())

# ✂️🌈 Slice those big PDF pages into bite-sized, overlapping chunks —
# 🧩 each piece shares a little context with its neighbours, so the retriever
# 🔍 never loses the thread when hunting for the perfect answer!

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(raw_doc)

# 🟣 Create embeddings, build the FAISS vector store and expose it as a similarity retriever

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding_model)
retriever = vector_store.as_retriever( search_type="similarity",search_kwargs={"k":3})

# 🟠 2. Re-initialize LLM with Chat Wrapper
# 🟠 LLM is served by Groq (GPT-OSS 120B)
llm = ChatGroq(model="openai/gpt-oss-120b", max_tokens=500)

# 🟡 3. Define the Prompt
rag_prompt = ChatPromptTemplate.from_template("""
Answer the question concisely, based only on the provided context.
If the answer is not in the context, reply exactly: "I don't know, it's not in my knowledge base."

<context>
{context}
</context>

Question: {input}""")

# 🔴 4. Construct Chain using LCEL
# 🔴 Helper: join retrieved chunks into a single context string

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 🔴 Pipeline: retrieve context -> fill prompt -> call LLM -> parse output to string

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 🟢 5. Confirm that everything was initialized
print("RAG Chain successfully initialized.")

# 📊 6. Show the vector DB (FAISS) as a pandas table, including each chunk's metadata
def vector_store_to_dataframe(store=vector_store, include_vectors=True):
    """One row per chunk: id, text, metadata columns (source, page...), and its embedding."""
    rows = []
    for faiss_idx, doc_id in store.index_to_docstore_id.items():
        doc = store.docstore.search(doc_id)
        row = {"faiss_index": faiss_idx, "doc_id": doc_id, "text": doc.page_content}
        row.update(doc.metadata)  # metadata -> separate columns (source, page, ...)
        if include_vectors:
            row["embedding"] = store.index.reconstruct(int(faiss_idx))
        rows.append(row)
    return pd.DataFrame(rows)


# 🧪 7. Test the RAG Chain end-to-end
# 🧪 Sample question used to verify retrieval + generation work together
if __name__ == "__main__":
    # 📋 Print the vector database as a table (full text/vectors truncated for readability)
    df = vector_store_to_dataframe()
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)
    print(f"Vector DB: {len(df)} chunks, embedding dim = {len(df['embedding'].iloc[0])}")
    print(df.drop(columns=["embedding"]).head(20).to_string())
    # df.to_csv("vector_db_table.csv", index=False)  # optional: export full table


    user_query = "What Is Rag?"

    # 🤖 Run the full pipeline: retrieve context -> build prompt -> call LLM -> parse text
    response = rag_chain.invoke(user_query)

    # 🖨️ Display the question and the model's generated answer
    print(f"Question: {user_query}")
    print(f"\nAnswer:\n{response}")