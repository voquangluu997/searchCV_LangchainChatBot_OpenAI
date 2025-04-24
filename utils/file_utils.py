import os
import shutil
import hashlib
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chainlit as cl
import tempfile
from chromadb.config import Settings

# config chunking
OPTIMAL_CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
CV_UPLOAD_DIR = "./data/uploaded_cvs"
CV_VECTOR_DB_DIR = "./data/cvs"

def get_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

async def process_uploaded_files(files: List[cl.File]) -> bool:
    """Xử lý và lưu trữ CV"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=OPTIMAL_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    all_docs = []
    temp_files = []
    try:
        for file in files:
            temp_path = None
            try:
                if not file.name.lower().endswith(".pdf"):
                    await cl.Message(content=f"File {file.name} is not PDF! Ignored...").send()
                    continue
                
                # Lưu file tạm để xử lý
                temp_path = os.path.join(tempfile.gettempdir(), f"temp_{file.name}")
                with open(temp_path, "wb") as f:
                    f.write(file.bytes if hasattr(file, 'bytes') else open(file.path, 'rb').read())
                temp_files.append(temp_path)
                
                # Lưu file chính thức
                save_path = os.path.join(CV_UPLOAD_DIR, file.name)
                with open(save_path, "wb") as f:
                    with open(temp_path, "rb") as temp:
                        f.write(temp.read())
                
                # Xử lý PDF
                loader = PyPDFLoader(temp_path)
                pages = loader.load_and_split(text_splitter)
                for page in pages:
                    page.metadata.update({
                        "source": file.name,
                        "page": page.metadata.get("page", 0) + 1
                    })
                all_docs.extend(pages)
                
                await cl.Message(content=f"✅ Processed {file.name}").send()

            except Exception as e:
                await cl.Message(content=f"❌ Error processing {file.name}: {str(e)}").send()
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                return False

        if not all_docs:
            await cl.Message(content="No valid CVs were processed!").send()
            return False
        
    finally:
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass
    return True
        

async def delete_cv_file(filename: str):
    """Xóa một CV cụ thể"""
    filepath = f"{CV_UPLOAD_DIR}/{filename}"
    if os.path.exists(filepath):
        os.remove(filepath)
    else:
        await cl.Message(content=f"❌ File '{filename}' not found.").send()
    
    # Xóa và tạo lại vector store với các file còn lại
    await rebuild_vector_store()

async def rebuild_vector_store():
    await clear_all_vector_db()
    os.makedirs(CV_UPLOAD_DIR, exist_ok=True)
    os.makedirs(CV_VECTOR_DB_DIR, exist_ok=True)
    cv_list = get_uploaded_cvs()
    try:
        if not cv_list:
            return None
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=OPTIMAL_CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True
        )
        all_docs = []
        for cv in cv_list:
            loader = PyPDFLoader(f"{CV_UPLOAD_DIR}/{cv}")
            pages = loader.load_and_split(text_splitter)
            for page in pages:
                page.metadata.update({
                    "source": cv,
                    "page": page.metadata.get("page", 0) + 1,
                    "original_file": cv
                })
            all_docs.extend(pages)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
        vectorstore = Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings
        )
        return vectorstore
        
    except Exception as e:
        print(f"Error rebuilding vector store: {str(e)}")
        return None

def get_uploaded_cvs() -> List[str]:
    if os.path.exists(CV_UPLOAD_DIR):
        return sorted([f for f in os.listdir(CV_UPLOAD_DIR) if f.lower().endswith(".pdf")])
    return []

async def clear_all_vector_db():
    """Xóa toàn bộ CV"""
    shutil.rmtree(CV_VECTOR_DB_DIR, ignore_errors=True)
    os.makedirs(CV_VECTOR_DB_DIR, exist_ok=True)
    os.chmod(CV_VECTOR_DB_DIR, 0o777)

async def clear_all_data():
    """Xóa toàn bộ CV"""
    await clear_all_vector_db()
    shutil.rmtree(CV_UPLOAD_DIR, ignore_errors=True)
    os.makedirs(CV_UPLOAD_DIR, exist_ok=True)
    os.chmod(CV_UPLOAD_DIR, 0o777)