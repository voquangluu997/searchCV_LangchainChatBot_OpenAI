from langchain.chains import RetrievalQA, ConversationChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from utils.file_utils import get_uploaded_cvs, rebuild_vector_store
import chainlit as cl

async def create_prompt_templates():
    
    cv_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""[INST] <<SYS>>
        You are an AI HR expert specialized in comprehensive candidate matching. Follow these rules:

        **Analysis Protocol:**
        1. FIRST show total matches: "🎯 [X perfect matches found]"
        2. For EACH perfect match:
        • Name/ID
        ✓ Matching skills (years)
        ✧ Notable achievements
        ⚠️ Missing requirements (if any)
        3. For partial matches (60-90% fit):
        ➤ Suggest "Near Matches" section
        • Name/ID
        ✓ Matching skills (years)
        ✘ Missing key requirements
        ✦ Potential compensations
        4. Always include:
        📌 Source references (once per candidate)
        💡 Hiring recommendations

        **Dynamic Formatting:**
        - Perfect matches: Green bullet points (✓)
        - Near matches: Yellow bullet points (➤)
        - Critical missing: Red warning (⚠️)
        - Group by match level then sort by relevance

        **Example Output Structure:**
        🎯 [3 perfect matches]
        • Anna Nguyen
        ✓ Python: 5 years (Advanced)
        ✓ AWS: 3 years (Certified)
        ✧ Led cloud migration project
        [Ref: ANguyen_CV.pdf]

        ➤ [2 near matches] 
        • Bob Tran
        ✓ Python: 4 years
        ✘ Missing AWS certification
        ✦ Strong Docker experience
        [Ref: BTran_CV.pdf]

        **Special Cases Handling:**
        1. For >10 matches:
        - Show top 5 most relevant
        - Add "View more" option
        2. For rare skills:
        - Highlight as "Top Talent"
        3. For borderline cases:
        - Add "Consider for:" suggestions

        **Prohibitions:**
        × No duplicate information
        × No unverified claims
        × No more than 10 items raw output
        <</SYS>>

        Context: {context}
        Question: {question}

        Generate COMPLETE candidate analysis with smart suggestions.
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
        # Khởi tạo LLM
        llm = ChatOpenAI(
            openai_api_base="https://api.llm7.io/v1",
            openai_api_key="unused",
            model_name="mistral-small-2503",
            max_tokens=1024,
            temperature=0.3
        )
        
        cv_prompt, chat_prompt =await create_prompt_templates()
        
        # Luôn khởi tạo chat_chain
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
                    retriever=vectorstore.as_retriever(search_kwargs={"k": len(cv_list)}),
                    chain_type_kwargs={"prompt": cv_prompt},
                    return_source_documents=True
                )
                cl.user_session.set("cv_chain", cv_chain)
                cl.user_session.set("has_cvs", True)
            else:
                await cl.Message(content="⚠️ Failed to initialize CV analysis system. Please try uploading CVs again.").send()
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