import os
import time
import faiss
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

def update_vector_store(folder_path: str, db_name: str, ollama_model: str = "nomic-embed-text", base_url="http://localhost:11434"):
    print(f"[{time.strftime('%X')}] กำลังโหลด vector store เดิม...")
    embeddings = OllamaEmbeddings(model=ollama_model, base_url=base_url)

    # ตรวจสอบว่ามีไฟล์ FAISS และ pkl อยู่หรือไม่
    faiss_path = os.path.join(db_name, "index.faiss")
    pkl_path = os.path.join(db_name, "index.pkl")

    if os.path.exists(faiss_path) and os.path.exists(pkl_path):
        print(f"[{time.strftime('%X')}] โหลด vector store เดิม...")
        try:
            vector_store = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"[{time.strftime('%X')}] ไม่สามารถโหลด vector store: {e}")
            vector_dim = len(embeddings.embed_query("Hello world"))
            index = faiss.IndexFlatL2(vector_dim)

            vector_store = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
    else:
        print(f"[{time.strftime('%X')}] ไม่พบ vector store เดิม สร้างใหม่...")
        vector_dim = len(embeddings.embed_query("Hello world"))
        index = faiss.IndexFlatL2(vector_dim)
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

    # ตรวจไฟล์ PDF ที่โหลดไว้แล้ว
    indexed_files = {
        doc.metadata.get("source"): doc.metadata.get("last_modified", 0)
        for doc in vector_store.docstore._dict.values()
    }

    # หาว่าไฟล์ไหนใหม่หรือถูกแก้ไข
    new_pdfs = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                path = os.path.join(root, file)
                mod_time = os.path.getmtime(path)
                if path not in indexed_files or mod_time > indexed_files[path]:
                    new_pdfs.append((path, mod_time))

    if not new_pdfs:
        print(f"[{time.strftime('%X')}] ไม่มีไฟล์ใหม่หรือไฟล์ที่ถูกแก้ไข")
        return

    # โหลดเอกสารใหม่
    docs = []
    for path, mod_time in new_pdfs:
        print(f"[{time.strftime('%X')}] โหลด: {path}")
        loader = PyMuPDFLoader(path)
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = path
            doc.metadata["last_modified"] = mod_time
        docs.extend(loaded_docs)

    # แบ่งข้อความ
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    if chunks:
        # ตรวจสอบว่า embeddings ถูกคำนวณแล้ว
        if len(chunks) > 0:
            print(f"[{time.strftime('%X')}] กำลังคำนวณเวกเตอร์...")
            vector_store.add_documents(chunks)
            vector_store.save_local(db_name)
            print(f"[{time.strftime('%X')}] บันทึก vector store เรียบร้อยที่ '{db_name}'")
        else:
            print(f"[{time.strftime('%X')}] ไม่พบข้อมูลในเอกสารใหม่")
    else:
        print(f"[{time.strftime('%X')}] ไม่มีเอกสารใหม่หรือข้อมูลให้เพิ่ม")

if __name__ == "__main__":
    update_vector_store(folder_path="rag-dataset", db_name="hr_info")
