import boto3
import json
import re

client = boto3.client("bedrock-runtime", region_name="us-east-2")

MODEL_ID = "us.amazon.nova-micro-v1:0"

system = [{ "text": "You are an AI assitant that can summarize, create names for documents, and come up with the document type." }]

def nova_micro_parser(text_content):
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

        parts = outputs.split("### ")

        summary, title, doc_type = "", "", ""
     
        for part in parts:
            if part.startswith("Summary"):
                summary = part.replace("Summary", "").strip().lstrip(":\n")
            elif part.startswith("Title"):
                title = part.replace("Title", "").strip().lstrip(":\n").strip("*").strip('"')
            elif part.startswith("Document Type"):
                doc_type = part.replace("Document Type", "").strip().lstrip(":\n").split("\n\n")[0].strip("*")


        return summary, title, doc_type
    
    except Exception as e:
        print("Error:", e)
        return "Error retrieving summary", "Error retrieving title"

def textract_parser(s3_bucket, s3_key):
    textract = boto3.client('textract', region_name='us-east-2')

    # get full text first
    full_response = textract.detect_document_text(
        Document={'S3Object': {'Bucket': s3_bucket, 'Name': s3_key}}
    )

    # append all text to a list
    extracted_text = " ".join([block.get("Text") for block in full_response.get("Blocks", []) if block.get("BlockType") == "LINE"])

    # define queries for textract
    queries = [
        {"Text": "What is the document type?"},
        {"Text": "If the expiration date exists, extract the expiration date in MM/DD/YYYY format."},
        {"Text": "If a U.S. state appears in this document, provide either the state name or state abbreviation."}
    ]

    # run textract based on queries
    response = textract.analyze_document(
        Document={'S3Object': {'Bucket': s3_bucket, 'Name': s3_key}},
        FeatureTypes=["QUERIES"],
        QueriesConfig={"Queries": queries}
    )

    # initialize results json format
    results = {
        "document_type": "",
        "title": "",
        "expiration_date": ["", "0.0"],
        "state": ["", "0.0"],
        "summary": "",
        "full_text": extracted_text
    }

    # look for query results
    for block in response.get('Blocks', []):
        if block.get('BlockType') == 'QUERY_RESULT':
            extracted_value = block.get('Text', "")
            confidence = block.get('Confidence', 0.0)

            if "document type" in block.get("Query", {}).get("Text", "").lower():
                results["document_type"] = [extracted_value, confidence]
            elif re.match(r"^\d{2}/\d{2}/\d{4}$", extracted_value):
                results["expiration_date"] = [extracted_value, confidence]
            elif re.match(r"^[A-Z]{2}$", extracted_value):
                results["state"] = [extracted_value, confidence]

    # call nova-lite for summary and title
    summary, title, doc_type = nova_micro_parser(extracted_text)
    results["summary"] = summary
    results["title"] = title
    results["document_type"] = doc_type

    return json.dumps(results, indent=4)

s3_bucket = "billiaitest"
s3_key = "diploma.jpg"

result = textract_parser(s3_bucket, s3_key)
print(result)
