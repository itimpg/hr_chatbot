import os
import re
import time
import faiss

from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

def clean_text(text):
    text = re.sub(r'\n', '\n', text)
    text = re.sub(r'\t+', '\t', text)
    text = re.sub(r'\t\s+', ' ', text)
    text = re.sub(r'\n\s+', '\n', text)
    text = text.strip()
    return text

driver = webdriver.Chrome() 

def scrape_website_content(url):
    driver.get(url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "main.entry-content"))
    )

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    raw_content = soup.find('main', {'class': 'entry-content'})
    content = clean_text(raw_content.get_text())
    return {"url": url, "content": content}

def store_in_vector_database(contents, db_name="hr_info", ollama_model="nomic-embed-text", base_url="http://localhost:11434"):
    embeddings = OllamaEmbeddings(model=ollama_model, base_url=base_url)

    faiss_path = os.path.join(db_name, "index.faiss")
    pkl_path = os.path.join(db_name, "index.pkl")

    if os.path.exists(faiss_path) and os.path.exists(pkl_path):
        vector_store = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)
    else:
        vector_dim = len(embeddings.embed_query("Hello world"))
        index = faiss.IndexFlatL2(vector_dim)
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

    documents = [Document(page_content=content["content"], metadata={"url": content["url"]}) for content in contents]  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    if len(chunks) > 0:
        print(f"[{time.strftime('%X')}] กำลังคำนวณเวกเตอร์...")
        vector_store.add_documents(chunks)
        vector_store.save_local(db_name)
        print(f"[{time.strftime('%X')}] บันทึก vector store เรียบร้อยที่ '{db_name}'")
    else:
        print(f"[{time.strftime('%X')}] ไม่พบข้อมูลในเอกสารใหม่")

def scrape_and_store(urls):
    scraped_contents = []
    for url in urls:
        try:
            content = scrape_website_content(url)
            scraped_contents.append(content)
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    if scraped_contents:
        store_in_vector_database(scraped_contents)

if __name__ == "__main__":
    website_urls = [
        "https://manaosoftware.com/about-us/",
        "https://manaosoftware.com/services/",
        "https://manaosoftware.com/outsourced-development-teams/",
        "https://manaosoftware.com/web-app-development-services/",
        "https://manaosoftware.com/mobile-app-development/",
        "https://manaosoftware.com/outsourced-software-testing/",
        "https://manaosoftware.com/web-application-penetration-testing-services/",
        "https://manaosoftware.com/services/artificial-intelligence/"
    ]
    scrape_and_store(website_urls)