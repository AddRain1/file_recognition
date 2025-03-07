import boto3
import json

def textract_parser(s3_bucket, s3_key):
    textract = boto3.client('textract', region_name='us-east-2')

    doc_type_query = [{"Text": "What is the document type closest to, is it a resume, diploma, certificate, driver's license, or other?"}]

    response = textract.analyze_document(
        Document={'S3Object': {'Bucket': s3_bucket, 'Name': s3_key}},
        FeatureTypes=["QUERIES"],
        QueriesConfig={"Queries": doc_type_query}
    )

    doc_type = "document" 
    doc_type_confidence = 0.0
    for block in response.get('Blocks', []):
        if block.get('BlockType') == 'QUERY_RESULT':
            print(block)
            doc_type = block.get('Text', "").lower()
            doc_type_confidence = block.get('Confidence', 0.0)

    # store document type and confidence
    results = [
        {
            "Text": doc_type_query[0]["Text"],
            "Answer": doc_type,
            "Confidence": doc_type_confidence
        }
    ]

    # generate queries based on document type
    queries = [
        {"Text": f"What is the name of this {doc_type}?"},
        {"Text": f"What is the expiration date of this {doc_type}?"},
        {"Text": f"What state is this {doc_type} from?"},
        {"Text": "Are there any important comments?"}
    ]

    # run textract with updated queries
    response = textract.analyze_document(
        Document={'S3Object': {'Bucket': s3_bucket, 'Name': s3_key}},
        FeatureTypes=["QUERIES"],
        QueriesConfig={"Queries": queries}
    )

    # store query results
    query_results = {query["Text"]: {"Answer": "", "Confidence": 0.0} for query in queries}

    for block in response.get('Blocks', []):
        if block.get('BlockType') == 'QUERY_RESULT':
            question_text = queries[len(query_results) - len(queries)]["Text"]
            query_results[question_text] = {
                "Answer": block.get('Text', ""),
                "Confidence": block.get('Confidence', 0.0)
            }

    # convert to json format
    for query in queries:
        results.append({
            "Text": query["Text"],
            "Answer": query_results[query["Text"]]["Answer"],
            "Confidence": query_results[query["Text"]]["Confidence"]
        })

    return json.dumps(results, indent=4)

s3_bucket = "billiaitest"
s3_key = "image.png"

result = textract_parser(s3_bucket, s3_key)
print(result)