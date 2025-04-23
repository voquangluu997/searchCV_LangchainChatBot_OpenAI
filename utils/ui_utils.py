import chainlit as cl
from typing import List
from utils.file_utils import get_uploaded_cvs 

async def display_cv_list():
    cv_list = get_uploaded_cvs()
    
    if cv_list:
        content = "📂 **Current CVs:**\n" + "\n".join(f"- {cv}" for cv in cv_list)
    else:
        content = "📂 No CVs uploaded yet"
    
    actions = create_management_actions(cv_list)
    await cl.Message(content=content, actions=actions).send()

def create_management_actions(cv_list: List[str]) -> List[cl.Action]:
    """Tạo các action buttons quản lý CV"""
    actions = [
        cl.Action(
            name="upload_cv",
            value="upload",
            label="📤 Upload New CV",
            description="Upload CV files for analysis",
            type="button",
            payload = {"value" : "upload_cv"}

        )
    ]
    
    if cv_list:
        actions.append(
            cl.Action(
                name="delete_all_cvs",
                value="all",
                label="🗑️ Delete All CVs",
                description="Remove all uploaded CVs",
                type="button",
                confirm=True,
                confirm_text="Are you sure you want to delete ALL CVs?",
            payload={"action": "delete_all"} 
            )
        )
        
        for cv in cv_list:
            actions.append(
                cl.Action(
                    name="delete_single_cv",
                    value=cv,
                    label=f"❌ Delete {cv}",
                    description=f"Remove this CV",
                    type="button",
                    confirm=True,
                    confirm_text=f"Are you sure you want to delete {cv}?",
                    payload={"action": "delete_single", "filename": cv} 

                )
            )
    
    return actions

async def show_file_upload_ui() -> List[cl.File]:
    """Hiển thị UI upload file và trả về danh sách file"""
    files = await cl.AskFileMessage(
        content="Upload CV PDFs (max 100 files)",
        accept=["application/pdf"],
        max_files=100,
        max_size_mb=100,
        timeout=600
    ).send()
    return files if files else []