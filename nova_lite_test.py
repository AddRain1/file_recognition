import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-east-2")

MODEL_ID = "us.amazon.nova-lite-v1:0"

system = [{ "text": "You are an AI assitant that can summarize and create names for documents." }]

def nova_lite_parser(text_content):
    messages = [
        {
            "role": "user",
            "content": [
                {"text": text_content},
                {"text": f"I want you to do 3 things: Summarize the document, generate a title for the document, and provide the document type."}
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
        outputs = model_response.get("output").get("message").get("content")[0].get("text")
        print(outputs)
        parts = outputs.split("**Generated Title:**\n")
        if len(parts) == 2:
            summary = parts[0].replace("**Summary:**\n", "").strip()
            title = parts[1].strip()
        else:
            summary = ""
            title = ""

        return summary, title
    
    except Exception as e:
        print("Error:", e)
        return "Error retrieving summary", "Error retrieving title"