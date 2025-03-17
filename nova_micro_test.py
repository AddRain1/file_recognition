import boto3
import json

# Initialize AWS Bedrock Runtime client
client = boto3.client("bedrock-runtime")

# Define Model ID for Amazon Nova Micro (Titan Text Micro)
MODEL_ID = "us.amazon.nova-micro-v1:0"

# Function to process text and extract summary, title, and document type
def process_document(text_content):
    messages = [
        {
            "role": "user",
            "content": [
                {"text": text_content},
                {"text": "I want you to do 3 things: Summarize the document, generate a title for the document, and provide the document type."}
            ],
        }
    ]

    inf_params = {
        "maxTokens": 200,  # Allow for more detailed responses
        "topP": 0.2,
        "topK": 20,
        "temperature": 0.5
    }

    body = json.dumps({
        "schemaVersion": "messages-v1",
        "messages": messages,
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

        parts = outputs.split("### ")

        summary, title, doc_type = "", "", ""
        print(parts)
        for part in parts:
            if part.startswith("Summary"):
                summary = part.replace("Summary", "").strip()
            elif part.startswith("Title"):
                title = part.replace("Title", "").strip().strip("*").strip('"')
            elif part.startswith("Document Type"):
                doc_type = part.replace("Document Type", "").strip().split("\n\n")[0].strip("*")

        return summary, title, doc_type
    
    except Exception as e:
        print("Error:", e)
        return "Error retrieving summary", "Error retrieving title", "Error retrieving document type"

# Example usage with extracted text
text_content = "This is a sample document about financial reports and Q1 earnings."
summary, title, doc_type = process_document(text_content)
print("Summary:", summary)
print("Title:", title)
print("Document Type:", doc_type)

# Define S3 bucket and file
s3_bucket = "billiaitest"
s3_key = "diploma.jpg"  # Supports PDF, JPG, PNG, etc.

