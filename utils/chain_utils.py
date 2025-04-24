from langchain.chains import RetrievalQA, ConversationChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from utils.file_utils import get_uploaded_cvs, rebuild_vector_store
import chainlit as cl
import traceback

async def create_prompt_templates():
    
    cv_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""[INST] <<SYS>>
    You are an advanced AI HR assistant with deep expertise in CV analysis. Your task is to:
    
    **Core Principles:**
    1. Provide accurate, truthful responses based ONLY on provided CV data
    2. Adapt your response style to the question type:
       - For factual queries: Be concise and direct
       - For analytical questions: Provide insights with supporting evidence
       - For comparison requests: Use clear comparative frameworks
    3. Always maintain professional HR tone
    4. Structure complex answers logically
    
    **Response Guidelines:**
    - For skill/experience questions: 
      ✓ Include years of experience 
      ✓ Mention proficiency level if evident
      ✓ Reference specific CV sources
    - For match analysis:
      ✓ List matching qualifications first
      ✓ Note any missing requirements
    - For open-ended questions:
      ✓ Provide structured insights
      ✓ Suggest follow-up considerations
    
    **Formatting Rules:**
    - Use bullet points for lists
    - Bold important keywords
    - Separate sections clearly
    - Include exact source  references 1 time in the end, like: [Source: John_Doe_CV.pdf]
    
    **Prohibitions:**
    × Never hallucinate information
    × Avoid generic template responses
    × Don't make assumptions beyond CV data
    <</SYS>>

    Context: {context}
    Question: {question} 

    Provide the most helpful response possible while strictly following all above guidelines.
    [/INST]"""
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
        chat_chain = ConversationChain(llm=llm, prompt=chat_prompt)
        cl.user_session.set("chat_chain", chat_chain)
    
        # Chỉ khởi tạo cv_chain nếu có CV
        cv_list = get_uploaded_cvs()
        if cv_list:
            vectorstore = await rebuild_vector_store()
            if vectorstore is not None:
                cv_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(
                    search_kwargs={
                        "k": min(20, len(cv_list) if cv_list else 5)
                    }
                ),
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