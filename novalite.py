import boto3
import json
import base64

client = boto3.client("bedrock-runtime", region_name="us-east-2")

MODEL_ID = "us.amazon.nova-lite-v1:0"

system = [{ "text": "You are an AI assitant that can summarize and create names for documents." }]

def send_request(text_content, request_type):
    messages = [
        {
            "role": "user",
            "content": [
                {"text": text_content},
                {"text": f"Please {request_type} the document."}
            ],
        }
    ]

    inf_params = {
        "maxTokens": 150, 
        "topP": 0.2, 
        "topK": 20, 
        "temperature": 0.5
    }

    body = json.dumps({
        "schemaVersion": "messages-v1",
        "messages": messages,
        "system": system,
        "inferenceConfig": inf_params,
    })

    try:
        response = client.invoke_model(
            modelId = MODEL_ID,
            body = body
        )

        model_response = json.loads(response["body"].read())
        return model_response.get("output").get("message").get("content")[0].get("text")

    except Exception as e:
        print("Error:", e)
        return ""

summary = send_request("summarize")
title = send_request("generate a title for")
print("Summary:", summary)
print("Title:", title)