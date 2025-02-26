import boto3
import json

def textract_parser(s3_bucket, s3_key, queries):
    textract = boto3.client('textract', region_name='us-east-2')

    # call Textract with queries
    response = textract.analyze_document(
        Document={'S3Object': {'Bucket': s3_bucket, 'Name': s3_key}},
        FeatureTypes=["QUERIES"],
        QueriesConfig={"Queries": queries}
    )

    query_results = []
    query_texts = [query["Text"] for query in queries]  # store questions in order

    # look for block with QUERY_RESULT type and store the answer
    for block in response.get('Blocks', []):
        if block.get('BlockType') == 'QUERY_RESULT':
            res_text = block.get('Text', "")
            query_results.append(res_text)

    # merge answers into the original query JSON format
    for i in range(len(queries)):
        queries[i]["Answer"] = query_results[i] if i < len(query_results) else ""

    return(json.dumps(queries, indent=4))

s3_bucket = "billiaitest"
s3_key = "image.png"

queries = [
        {"Text": "What is the document type?"},
        {"Text": "What is the name of the document?"},
        {"Text": "What is the expiration date?"},
        {"Text": "What state is the document from?"},
        {"Text": "Are there any important comments?"}
]

result = textract_parser(s3_bucket, s3_key, queries)
print(result)