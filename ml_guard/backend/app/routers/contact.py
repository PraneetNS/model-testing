from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class ContactQuery(BaseModel):
    name: str
    email: EmailStr
    message: str

def send_email_mock(query: ContactQuery):
    # In a real scenario, use an email client like SendGrid or SMTP
    logger.info(f"EMAIL SENT TO savantpraneet@gamil.com:")
    logger.info(f"From: {query.name} ({query.email})")
    logger.info(f"Message: {query.message}")
    # Simulating successful email dispatch
    return True

@router.post("/contact")
async def contact_us(query: ContactQuery, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(send_email_mock, query)
        return {"status": "success", "message": "Query received and forwarded to savantpraneet@gamil.com"}
    except Exception as e:
        logger.error(f"Contact form error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process request")
