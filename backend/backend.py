from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()  # load HF_TOKEN / GROQ_API_KEY from .env
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import pandas as pd

from config import KNOWLEDGE_BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME
from ingest import sync_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")



# 🔵 1. Re-initialize Retriever
# 🟢 Every PDF in the "knowledge base" folder (one level above /backend) is indexed

# ✂️🌈 Slice those big PDF pages into bite-sized, overlapping chunks —
# 🧩 each piece shares a little context with its neighbours, so the retriever
# 🔍 never loses the thread when hunting for the perfect answer!

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# 🟣 Create embeddings and the FAISS vector store, then expose it as a similarity retriever
# ⚡ Incremental: the index is saved to disk and only NEW / CHANGED PDFs are
# ⚡ loaded, split and embedded. Removed PDFs are dropped. See ingest.py.

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_store = sync_vector_store(KNOWLEDGE_BASE_DIR, splitter, embedding_model)
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