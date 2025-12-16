from fastapi import FastAPI, Request
from typing import Optional
import requests
import os
import openai
from rag import get_rag_chain

rag_chain = get_rag_chain()
app = FastAPI()

VERIFY_TOKEN = "verify_token_123"

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# at the top of ingest.py and rag.py
from dotenv import load_dotenv
load_dotenv()
openai.api_key = OPENAI_API_KEY


@app.get("/webhook")
def verify_webhook(
    hub_mode: Optional[str] = None,
    hub_challenge: Optional[str] = None,
    hub_verify_token: Optional[str] = None,
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN and hub_challenge is not None:
        return int(hub_challenge)
    return "Verification failed"


@app.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()

    try:
        message = data["entry"][0]["changes"][0]["value"]["messages"][0]
        sender = message["from"]
        user_text = message["text"]["body"]

        reply = ai_reply(user_text)
        send_whatsapp_message(sender, reply)

    except Exception as e:
        print("Error:", e)

    return {"status": "ok"}




def ai_reply(user_message: str) -> str:
    result = rag_chain.invoke(user_message)
    return result["result"]




def send_whatsapp_message(to: str, text: str):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }

    requests.post(url, headers=headers, json=payload)
