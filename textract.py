import boto3
import json
import re
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

def textract_parser(s3_bucket, s3_key):
    textract = boto3.client('textract', region_name='us-east-2')

    # Step 1: Identify Document Type
    doc_type_query = [
        {"Text": "What is the document type closest to, is it a resume, diploma, certificate, driver's license, or other?"}
    ]

    full_response = textract.analyze_document(
        Document={'S3Object': {'Bucket': s3_bucket, 'Name': s3_key}}
    )

    extracted_text = []
    for block in full_response["Blocks"]:
        if block["BlockType"] == "LINE":
            extracted_text.append(block["Text"])

    response = textract.analyze_document(
        Document={'S3Object': {'Bucket': s3_bucket, 'Name': s3_key}},
        FeatureTypes=["QUERIES"],
        QueriesConfig={"Queries": doc_type_query}
    )

    doc_type = "document" 
    doc_type_confidence = 0.0
    for block in response.get('Blocks', []):
        if block.get('BlockType') == 'QUERY_RESULT':
            doc_type = block.get('Text', "").lower()
            doc_type_confidence = block.get('Confidence', 0.0)

    # Step 2: Define Queries Based on Document Type
    queries = [
        {"Text": f"If the expiration date exists, extract the expiration date in MM/DD/YYYY format for this {doc_type}."},
        {"Text": f"If a U.S. state appears in this {doc_type} document, provide either the state name or state abbreviation."},
    ]

    # Step 3: Run Textract Again with New Queries
    response2 = textract.analyze_document(
        Document={'S3Object': {'Bucket': s3_bucket, 'Name': s3_key}},
        FeatureTypes=["QUERIES"],
        QueriesConfig={"Queries": queries}
    )

    # Step 4: Initialize Output Dictionary
    results = {
        "document_type": [doc_type, doc_type_confidence],
        "expiration_date": ["", "0.0"],
        "state": ["", "0.0"],
        "full_text": extracted_text
    }

    # Step 5: Extract Query Results
    for block in response2.get('Blocks', []):
        if block.get('BlockType') == 'QUERY_RESULT':
            print(block)
            extracted_value = block.get('Text', "")
            confidence = block.get('Confidence', 0.0)

            if re.match(r"^\d{2}/\d{2}/\d{4}$", extracted_value): 
                results["expiration_date"] = [extracted_value, confidence]
            elif re.match(r"^[A-Z]{2}$", extracted_value):
                results["state"] = [extracted_value, confidence]

    return json.dumps(results, indent=4)

s3_bucket = "billiaitest"
s3_key = "diploma.jpg"

summary = send_request(base64_string, "summarize")
title = send_request(base64_string, "generate a title for")

result = textract_parser(s3_bucket, s3_key)
print(result)
