import chainlit as cl
from dotenv import load_dotenv
import os
from utils.file_utils import (
    process_uploaded_files,
    delete_cv_file,
    clear_all_data
)
from utils.ui_utils import (
    display_cv_list
)
from utils.chain_utils import initialize_chains, is_cv_related_question, initialize_chains

load_dotenv()

# Khởi tạo thư mục
os.makedirs("./data/uploaded_cvs", exist_ok=True)
os.makedirs("./data/cvs", exist_ok=True)



@cl.on_chat_start
async def start_chat():
    await display_cv_list()
    await initialize_chains()
    await cl.Message(content="Welcome to HR Assistant! Use the buttons below to manage CVs.").send()

@cl.on_message
async def handle_message(message: cl.Message):
    user_input = message.content
    
    # Kiểm tra nếu là lệnh quản lý CV
    if user_input.lower() in ["refresh", "reload", "cv", "manager cv"]:
        await display_cv_list()
        return
    
    cv_chain = cl.user_session.get("cv_chain")
    chat_chain = cl.user_session.get("chat_chain")
    has_cvs = cl.user_session.get("has_cvs", False)
    
    try:
        if has_cvs and cv_chain is not None and is_cv_related_question(user_input):
            # Xử lý câu hỏi liên quan CV
            res = await cv_chain.ainvoke(
                {"query": user_input},
                callbacks=[cl.AsyncLangchainCallbackHandler()]
            )
            
            answer = res["result"]
            sources = {os.path.basename(doc.metadata["source"]) for doc in res["source_documents"]}
            response = f"📄 CV Analysis:\n{answer}\n\n🔍 Sources:\n" + "\n".join(f"- {s}" for s in sources)
        else:
            # Xử lý chat thông thường
            res = await chat_chain.ainvoke(
                {"input": user_input},
                callbacks=[cl.AsyncLangchainCallbackHandler()]
            )
            response = f"💬 {res['response']}"
        
        if not has_cvs and is_cv_related_question(user_input):
            response += "\n\n⚠️ Note: No CVs uploaded yet. Please upload CVs for detailed analysis."
        
        await cl.Message(content=response).send()
    
    except Exception as e:
        await cl.Message(content=f"⚠️ Error: {str(e)}").send()

@cl.action_callback("upload_cv")
async def on_upload(action: cl.Action):
    try:
        files = await cl.AskFileMessage(
            content="Please upload CV PDFs",
            accept=["application/pdf"],
            max_files=50,
            max_size_mb=100,
            timeout=300
        ).send()

        if not files:
            await cl.Message(content="No files were uploaded").send()
            return
        success = await process_uploaded_files(files)
        if success:
            await initialize_chains()
            await display_cv_list()
            await cl.Message(content="✅ CVs uploaded successfully!").send()
    except Exception as e:
        await cl.Message(content=f"❌ Upload failed: {str(e)}").send()

# @cl.action_callback("delete_single_cv")
# async def on_delete_single_cv(action: cl.Action):
#     try:
#         cv_name = action.payload.get("filename")
#         if not cv_name:
#             raise ValueError("Filename not provided in payload")
            
#         await delete_cv_file(cv_name)
#         await cl.Message(content=f"🗑️ CV '{cv_name}' deleted").send()
#         await initialize_chains()
#         await display_cv_list()
#     except Exception as e:
#         await cl.Message(content=f"❌ Error deleting CV: {str(e)}").send()

@cl.action_callback("delete_single_cv")
@cl.action_callback("delete_all_cvs")
async def on_delete(action: cl.Action):
    try: 
        action_type = action.payload.get("action")
        if action_type == "delete_all":
            cv_name = "all"
            await clear_all_data()
            await cl.Message(content=f"🗑️ All CV deleted").send()

        elif action_type == "delete_single":
            cv_name = action.payload.get("filename")
            if not cv_name:  
                raise ValueError("Filename not provided in payload")
            await delete_cv_file(cv_name)
            await cl.Message(content=f"🗑️ CV '{cv_name}' deleted").send()
        else:
            raise ValueError("Unknown delete action")

        await initialize_chains()
        await display_cv_list()
    except Exception as e:
        await cl.Message(content=f"❌ Error deleting CV: {str(e)}").send()

@cl.action_callback("open_cv")
async def on_open_cv(action: cl.Action):
    try:
        cv_name = action.payload.get("filename")
        if not cv_name:
            raise ValueError("Filename not provided in payload")
        file_path = f"./data/uploaded_cvs/{cv_name}"
        elements = [
            cl.Pdf(name=cv_name, content = cv_name, display="page", path=file_path)
        ]
        await cl.Message(content = f"View {cv_name}", elements = elements).send()
    except Exception as e:
        await cl.Message(content=f"❌ Error deleting CV: {str(e)}").send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit("main.py")
    
