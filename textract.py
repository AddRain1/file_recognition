import boto3
import json
import re

def textract_parser(s3_bucket, s3_key):
    textract = boto3.client('textract', region_name='us-east-2')

    # Step 1: Identify Document Type
    doc_type_query = [
        #{"Text": "Identify the document type by comparing it to common formats such as: Resume, Diploma, Certificate, Driver's License, Passport, Identification Card, or Other. Respond with only the document type."}
        {"Text": "What is the document type closest to, is it a resume, diploma, certificate, driver's license, or other?"}
    ]

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
        {"Text": f"Give me a name for this {doc_type} document."},
        {"Text": f"If the expiration date exists, extract the expiration date in MM/DD/YYYY format for this {doc_type}."},
        {"Text": f"If a U.S. state appears in this {doc_type} document, provide either the state name or state abbreviation."},
        {"Text": f"Summarize the key details of this {doc_type} document in one or two sentences."}
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
        "name": ["", "0.0"],
        "expiration_date": ["", "0.0"],
        "state": ["", "0.0"],
        "summary": ["", "0.0"]
    }

    # Step 5: Extract Query Results
    for block in response2.get('Blocks', []):
        if block.get('BlockType') == 'QUERY_RESULT':
            print(block)
            extracted_value = block.get('Text', "")
            confidence = block.get('Confidence', 0.0)

            if re.match(r"^[A-Za-z]+\s[A-Za-z]+$", extracted_value):
                results["name"] = [extracted_value, confidence]
            elif re.match(r"^\d{2}/\d{2}/\d{4}$", extracted_value): 
                results["expiration_date"] = [extracted_value, confidence]
            elif re.match(r"^[A-Z]{2}$", extracted_value):
                results["state"] = [extracted_value, confidence]
            else:
                results["summary"] = [extracted_value, confidence]

    return json.dumps(results, indent=4)

s3_bucket = "billiaitest"
s3_key = "image.png"

result = textract_parser(s3_bucket, s3_key)
print(result)
