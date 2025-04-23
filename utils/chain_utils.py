from langchain.chains import RetrievalQA, ConversationChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from utils.file_utils import get_uploaded_cvs, rebuild_vector_store
import chainlit as cl
import traceback
from langchain_chroma import Chroma

async def initialize_embeddings():
    """Khởi tạo embeddings model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

async def create_prompt_templates():
    """Tạo các prompt templates"""
    cv_prompt = PromptTemplate(
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
        You're a friendly and helpful AI assistant. Respond conversationally.
        History: {history}
        <</SYS>>
        
        {input} [/INST]"""
    )
    
    return cv_prompt, chat_prompt

async def initialize_chains():
    try:

        llm = ChatOpenAI(
                openai_api_base="https://api.llm7.io/v1",
                openai_api_key="unused",
                model_name="mistral-small-2503",
                max_tokens=1024,
                temperature=0.3,
                request_timeout=30
            )
        cv_prompt, chat_prompt = await create_prompt_templates()
        chat_chain = ConversationChain(llm=llm, prompt=chat_prompt)
        cl.user_session.set("chat_chain", chat_chain)

        # Load vector store
        cv_list = get_uploaded_cvs()
        if cv_list:
            embeddings = await initialize_embeddings()
            vectorstore = Chroma(
                persist_directory="./data/cvs",
                embedding_function=embeddings
            )
            cv_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                        chain_type_kwargs={"prompt": cv_prompt},
                        return_source_documents=True
                    )
            cl.user_session.set("cv_chain", cv_chain)
            cl.user_session.set("has_cvs", True)
        else:
            cl.user_session.set("cv_chain", None)
            cl.user_session.set("has_cvs", False)
    except Exception as e:
        print(f"Error rebuilding vector store: {str(e)}")
        traceback.print_exc() 
        return None

async def rebuild_cv_chain():
    try:
        # init LLM
        llm = ChatOpenAI(
            openai_api_base="https://api.llm7.io/v1",
            openai_api_key="unused",
            model_name="mistral-small-2503",
            max_tokens=1024,
            temperature=0.3,
            request_timeout=30
        )
        
        cv_prompt, chat_prompt = await create_prompt_templates()
    
        # Chỉ khởi tạo cv_chain nếu có CV
        cv_list = get_uploaded_cvs()
        if cv_list:
            vectorstore = await rebuild_vector_store()
            if vectorstore is not None:
                cv_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                    chain_type_kwargs={"prompt": cv_prompt},
                    return_source_documents=True
                )
                cl.user_session.set("cv_chain", cv_chain)
                cl.user_session.set("has_cvs", True)
            else:
                await cl.Message(content="⚠️ Failed to rebuild vector store.").send()
                cl.user_session.set("cv_chain", None)
                cl.user_session.set("has_cvs", False)
        else:
            cl.user_session.set("cv_chain", None)
            cl.user_session.set("has_cvs", False)
            
    except Exception as e:
        print(f"Error initializing chains: {str(e)}")
        await cl.Message(content="⚠️ Error initializing chatbot. Please try again.").send()
        cl.user_session.set("chat_chain", None)
        cl.user_session.set("cv_chain", None)
        cl.user_session.set("has_cvs", False)
    
def is_cv_related_question(question: str) -> bool:
    """Phân loại câu hỏi liên quan đến CV"""
    keywords = ["cv", "resume", "candidate", "experience", "skill", "ứng viên", "kỹ năng", "kinh nghiệm", "uv", "ưv"]
    return any(keyword in question.lower() for keyword in keywords)