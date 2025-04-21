import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA, ConversationChain
from langchain.prompts import PromptTemplate

from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
# from huggingface_hub import login
from langchain_openai import ChatOpenAI

load_dotenv()

# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# login(token=HUGGINGFACEHUB_API_TOKEN)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""[INST] <<SYS>>
You're an expert HR assistant. Analyze CVs and:
- List candidates matching ALL requirements
- Include years of experience for each skill
- Never invent information
- Format response with bullet points
- Reference source file names
<</SYS>>

Context: {context}
Question: {question} [/INST]"""
)

chat_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""[INST] <<SYS>>
You're a friendly and helpful AI assistant. Respond to the user in a natural conversational manner.
History: {history}
<</SYS>>

{input} [/INST]"""
)

# Classify questions related to CV.
def is_cv_related(question: str) -> bool:
    keywords = ["cv", "resume", "candidate", "experience", "skill", "ứng viên", "kỹ năng", "kinh nghiệm"]
    return any(keyword in question.lower() for keyword in keywords)

@cl.on_chat_start
async def init():
    files = None
    while files is None:
    # Processing PDF upload
        files = await cl.AskFileMessage(
            content="Please upload CV PDFs to start",
            accept=["application/pdf"],
            max_files=10,
            max_size_mb=50,
            timeout=300
        ).send()

        if not files:
            await cl.Message(content="You haven't uploaded a PDF file! Please try again.").send()
            files = None
            continue

    # Read and chunk PDF
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    all_docs = []
    for file in files:
        try:
            if not file.name.lower().endswith(".pdf"):
                await cl.Message(content=f"File {file.name} is not PDF! Ignored...").send()
                continue
            loader = PyPDFLoader(file.path)
            pages = loader.load_and_split(text_splitter)
            for page in pages:
                page.metadata.update({
                    "source": file.name,
                    "page": page.metadata.get("page", 0) + 1
                })
            all_docs.extend(pages)
            await cl.Message(content=f"Processed {file.name}).").send()

        
        except Exception as e:
            await cl.Message(content=f"Error when read file {file.name}: {str(e)}").send()
            continue

    if not all_docs:
        await cl.Message(content="No valid CV found! Please upload again.").send()
        return


    # Init Chroma
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory="./data/cvs",
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    llm = ChatOpenAI(
        openai_api_base="https://api.llm7.io/v1",
        openai_api_key="unused",
        model_name="mistral-small-2503",
        max_tokens=1024,
        temperature=0.3
    )
    
    cv_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )
    chat_chain = ConversationChain(llm=llm, prompt=chat_prompt)

    cl.user_session.set("cv_chain", cv_chain)
    cl.user_session.set("chat_chain", chat_chain)
    cl.user_session.set("has_cvs", bool(all_docs))    
    await cl.Message(content=f"Ready! Loaded {len(files)} CVs").send()

@cl.on_message
async def main(message: cl.Message):
    user_input = message.content
    cv_chain = cl.user_session.get("cv_chain")
    chat_chain = cl.user_session.get("chat_chain")
    has_cvs = cl.user_session.get("has_cvs")

    if has_cvs and is_cv_related(user_input):
        res = await cv_chain.ainvoke(
            {"query": user_input},
            callbacks=[cl.AsyncLangchainCallbackHandler()]
        )
        
        answer = res["result"]
        sources = {os.path.basename(doc.metadata["source"]) for doc in res["source_documents"]}
        response = f"📄 CV Analysis:\n{answer}\n\n🔍 Sources:\n" + "\n".join(f"- {s}" for s in sources)
    else:
        res = await chat_chain.ainvoke(
            {"input": user_input},
            callbacks=[cl.AsyncLangchainCallbackHandler()]
        )
        response = f"💬 Chat Response:\n{res['response']}"
    
    if not has_cvs and is_cv_related(user_input):
        response += "\n\n⚠️ Note: No CVs uploaded yet. Please upload CVs for detailed analysis."
    
    await cl.Message(content=response).send()
    
   