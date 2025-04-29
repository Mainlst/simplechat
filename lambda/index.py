# index.py – Lambda からローカル FastAPI (Gemma-2) を呼び出すサンプル

import json
import os
import logging
import requests
from typing import List, Dict, Any

# ---------- 設定 ----------
LLM_API_URL  = os.environ["LLM_API_URL"]  # ← 環境変数から読み出す
LLM_API_USER = os.environ.get("LLM_API_USER", "")
LLM_API_PASS = os.environ.get("LLM_API_PASS", "")
REQUEST_TIMEOUT = int(os.environ.get("LLM_API_TIMEOUT", "30"))
# --------------------------

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------- 共通ユーティリティ ----------
def _parse_event(event: Dict[str, Any]) -> Dict[str, Any]:
    if "body" not in event:
        raise ValueError("No body in event")
    body_raw = event["body"]
    if isinstance(body_raw, str):
        return json.loads(body_raw)
    return body_raw

def _call_local_llm(prompt: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    payload = {
        "prompt": prompt,
        "conversationHistory": history,
        "max_new_tokens": 512
    }
    auth = (LLM_API_USER, LLM_API_PASS) if LLM_API_USER else None
    logger.info("POST %s (timeout=%ss)", LLM_API_URL, REQUEST_TIMEOUT)
    resp = requests.post(
        LLM_API_URL,
        json=payload,
        auth=auth,
        timeout=REQUEST_TIMEOUT,
        headers={"Content-Type": "application/json"}
    )
    resp.raise_for_status()
    return resp.json()
# -----------------------------------------

def lambda_handler(event, context):
    try:
        req_body = _parse_event(event)
        prompt   = req_body["message"]
        history  = req_body.get("conversationHistory", [])

        llm_resp = _call_local_llm(prompt, history)
        assistant_response = llm_resp["generated_text"]
        updated_history = llm_resp.get(
            "conversationHistory",
            history + [{"role": "assistant", "content": assistant_response}]
        )

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": True,
                "response": assistant_response,
                "conversationHistory": updated_history
            })
        }

    except Exception as e:
        logger.exception("Error during inference")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(e)
            })
        }

