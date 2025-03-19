import boto3
import json
import uuid  

s3_bucket = "billiaitest"
s3_key = "image.png"

bedrock = boto3.client("bedrock-runtime")

def summarize_document(bucket, key):
    s3_uri = f"s3://{bucket}/{key}"

    tool_use_id = str(uuid.uuid4())

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolUse": {
                        "name": "document",
                        "toolUseId": tool_use_id,
                        "input": {"s3Uri": s3_uri}
                    }
                },
                {"text": "Summarize the contents of this document."}
            ]
        }
    ]

    response = bedrock.invoke_model(
        modelId="us.amazon.nova-lite-v1:0", 
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"messages": messages})
    )

    response_body = json.loads(response['body'].read())
    print(response_body)
    summary = response_body['output']['message']['content'][0]['text']

    return summary

summary = summarize_document(s3_bucket, s3_key)
print("Summary of the document:")
print(summary)
