import boto3
import json
import re
# import pymupdf

client = boto3.client("bedrock-runtime", region_name="us-east-2")

MODEL_ID = "us.amazon.nova-micro-v1:0"

system = [{"text": "You are an AI assistant that provides only JSON formatted responses. Do not include any extra text, just return the JSON object."}]

def normalize_json_keys(json_data):
    normalized_data = {}
    for key, value in json_data.items():
        normalized_key = key.replace(' ', '')
        normalized_data[normalized_key] = value
    return normalized_data

def nova_micro_parser(text_content):
    messages = [
        {
            "role": "user",
            "content": [
                {"text": text_content},
                {"text": f"Provide only the following information in JSON format: Summary, Title, and Document Type."}
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

        parsed_json = json.loads(outputs)
        parsed_json = normalize_json_keys(parsed_json)
        return parsed_json
    
    except Exception as e:
        print("Error:", e)
        return "Error retrieving summary", "Error retrieving title"

def textract_parser(s3_bucket, s3_key):
    textract = boto3.client('textract', region_name='us-east-2')

    # define queries for textract
    queries = [
        {"Text": "What is the expiration date of this document?"},
        {"Text": "What state is this document from?"}
    ]

    # run textract based on queries
    response = textract.analyze_document(
        Document={'S3Object': {'Bucket': s3_bucket, 'Name': s3_key}},
        FeatureTypes=["QUERIES"],
        QueriesConfig={"Queries": queries}
    )

    extracted_text = " ".join([block.get("Text") for block in response.get("Blocks", []) if block.get("BlockType") == "LINE"])

    # initialize results json format
    results = {
        "title": "",
        "document_type": "",
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

            if re.match(r"\d{2}/\d{2}/\d{4}", extracted_value):
                results["expiration_date"] = [extracted_value, confidence]
            elif re.match(r"[A-Z]{2}", extracted_value) or re.match(r"[A-Z][a-z]+", extracted_value):
                results["state"] = [extracted_value, confidence]

    # call nova micro for summary and title and docuemnt type
    nova_results = nova_micro_parser(extracted_text)
    results["summary"] = nova_results.get("Summary", "")
    results["title"] = nova_results.get("Title", "")
    results["document_type"] = nova_results.get("DocumentType", "")

    return json.dumps(results, indent=4)

s3_bucket = "billiaitest"
s3_key = "CPR Card.png"  # replace with the key of the document you want to analyze
# s3_key = "DD214-Example_Redacted_0.pdf"

result = textract_parser(s3_bucket, s3_key)
print(result)
