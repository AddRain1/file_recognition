import boto3
import json

client = boto3.client("bedrock-runtime")

MODEL_ID = "us.amazon.nova-micro-v1:0"

system = [{"text": "You are an AI assistant that provides only JSON formatted responses. Do not include any extra text, just return the JSON object."}]

def process_document(text_content):
    messages = [
        {
            "role": "user",
            "content": [
                {"text": text_content},
                {"text": "Provide only the following information in JSON format: Summary, Title, and Document Type."}
            ],
        }
    ]

    inf_params = {
        "maxTokens": 200, 
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
            modelId=MODEL_ID,
            body=body
        )

        # Parse the response
        model_response = json.loads(response["body"].read())
        outputs = model_response.get("output").get("message").get("content")[0].get("text")
        print("outputs:", outputs)
        try:
            parsed_json = json.loads(outputs)
            return parsed_json
        except json.JSONDecodeError:
            print("Error parsing JSON response:", outputs)
            return None
    
    except Exception as e:
        print("Error:", e)
        return "Error retrieving summary", "Error retrieving title", "Error retrieving document type"

# Example usage with extracted text
text_content = "This is a sample document about financial reports and Q1 earnings."
result = process_document(text_content)
print(result)


