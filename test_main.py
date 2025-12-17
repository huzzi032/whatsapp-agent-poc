import pytest
from fastapi.testclient import TestClient
from main import app, ai_reply, VERIFY_TOKEN
import os


client = TestClient(app)


def test_verify_webhook_success():
    """Test successful webhook verification."""
    response = client.get("/webhook", params={
        "hub_mode": "subscribe",
        "hub_challenge": "123456789",
        "hub_verify_token": VERIFY_TOKEN
    })

    assert response.status_code == 200
    assert response.json() == 123456789


def test_verify_webhook_failure():
    """Test webhook verification failure."""
    # Wrong mode
    response = client.get("/webhook", params={
        "hub_mode": "invalid",
        "hub_challenge": "123456789",
        "hub_verify_token": VERIFY_TOKEN
    })
    assert response.status_code == 200
    assert response.json() == "Verification failed"

    # Wrong token
    response = client.get("/webhook", params={
        "hub_mode": "subscribe",
        "hub_challenge": "123456789",
        "hub_verify_token": "wrong_token"
    })
    assert response.status_code == 200
    assert response.json() == "Verification failed"


def test_ai_reply():
    """Test that ai_reply returns a string response."""
    test_message = "What shoes do you have?"
    response = ai_reply(test_message)

    assert isinstance(response, str)
    assert len(response.strip()) > 0

    print(f"Test message: {test_message}")
    print(f"AI response: {response}")


def test_webhook_post():
    """Test the POST webhook endpoint with mock data."""
    # Mock WhatsApp message payload
    mock_payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "from": "1234567890",
                                    "text": {"body": "Hello"}
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

    # Note: This will try to send a WhatsApp message, but since we don't have real tokens,
    # it might fail, but we can check that the endpoint doesn't crash
    response = client.post("/webhook", json=mock_payload)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


if __name__ == "__main__":
    test_verify_webhook_success()
    test_verify_webhook_failure()
    test_ai_reply()
    test_webhook_post()
    print("All main.py tests passed!")