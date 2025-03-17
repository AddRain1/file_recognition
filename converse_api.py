import boto3
import json
import uuid  # To generate a unique toolUseId

# Define S3 bucket and document key (supports PDF, PNG, JPG, etc.)
s3_bucket = "billiaitest"
s3_key = "image.png"  # Change to your document file

# Initialize AWS Bedrock Runtime client
bedrock = boto3.client("bedrock-runtime")

# Function to summarize a document using Nova Lite Converse API
def summarize_document(bucket, key):
    s3_uri = f"s3://{bucket}/{key}"

    # Generate a unique toolUseId
    tool_use_id = str(uuid.uuid4())

    # Construct the Converse API request with required `toolUseId`
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolUse": {
                        "name": "document",
                        "toolUseId": tool_use_id,  # ✅ REQUIRED KEY
                        "input": {"s3Uri": s3_uri}
                    }
                },
                {"text": "Summarize the contents of this document."}
            ]
        }
    ]

    # Send request to Nova Lite via Converse API
    response = bedrock.invoke_model(
        modelId="us.amazon.nova-lite-v1:0",  # Nova Lite model
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"messages": messages})
    )

    # Parse the response
    response_body = json.loads(response['body'].read())
    print(response_body)
    summary = response_body['output']['message']['content'][0]['text']

    return summary

# Execute the function and print the summary
summary = summarize_document(s3_bucket, s3_key)
print("Summary of the document:")
print(summary)
