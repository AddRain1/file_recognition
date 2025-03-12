import boto3
import json

# AWS Clients
textract_client = boto3.client("textract", region_name="us-east-2")  # Textract for OCR
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-2")  # Nova Lite for summarization

MODEL_ID = "us.amazon.nova-lite-v1:0"

system = [{ "text": "You are an AI assistant that can summarize and create names for documents." }]

# Define S3 bucket and object key
s3_bucket = "billiaitest"  # Replace with your actual bucket name
s3_key = "diploma.jpg"  # Replace with the actual S3 object key

def extract_text_from_image_s3(bucket, key):
    """Uses Amazon Textract to extract text from an image in S3."""
    try:
        response = textract_client.analyze_document(
            Document={"S3Object": {"Bucket": bucket, "Name": key}},
            FeatureTypes=["TABLES", "FORMS"]
        )
        
        extracted_text = []
        for block in response["Blocks"]:
            if block["BlockType"] == "LINE":
                extracted_text.append(block["Text"])
        
        return "\n".join(extracted_text)
    
    except Exception as e:
        print("Error in Textract OCR:", e)
        return ""

def send_request_nova(text_content, request_type):
    """Send extracted text to Amazon Nova Lite for summarization or title generation."""
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
        response = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=body
        )

        model_response = json.loads(response["body"].read())
        return model_response.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
    
    except Exception as e:
        print("Error in Nova Lite API:", e)
        return ""

# **Step 1:** Extract text using Amazon Textract
extracted_text = extract_text_from_image_s3(s3_bucket, s3_key)

if extracted_text:
    # **Step 2:** Get the summary
    summary = send_request_nova(extracted_text, "summarize")
    print("\nSummary:\n", summary)

    # **Step 3:** Get the title
    title = send_request_nova(extracted_text, "generate a suitable title for")
    print("\nTitle:\n", title)
else:
    print("No text extracted from image.")
