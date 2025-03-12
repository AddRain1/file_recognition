import boto3
import json

# Initialize the S3 client
s3_client = boto3.client("s3")

# Specify the bucket name and image file name
bucket_name = "billiaitest"
image_key = "image.png" 

# Generate a pre-signed URL that expires in 3600 seconds (1 hour)
presigned_url = s3_client.generate_presigned_url(
    "get_object",
    Params={"Bucket": bucket_name, "Key": image_key},
    ExpiresIn=3600,
)

print("Pre-Signed URL:", presigned_url)

client = boto3.client("bedrock-runtime", region_name="us-east-2")

MODEL_ID = "us.amazon.nova-lite-v1:0"

system = [{ "text": "You are an AI assitant that can summarize and create names for documents." }]

messages = [
    {
        "role": "user",
        "content": [
            {
                "image": {
                    "format": "png",
                    "source": {"url": presigned_url},
                }
            },
            {
                "text": "Please summarize the document and suggest a suitable title."
            }
        ],
    }
]

inf_params = {
    "maxTokens": 500, 
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
    response = client.invoke_model (
        modelId = MODEL_ID,
        body = body,
        accept = "application/json",
        contentType = "application/json"
    )

    model_response = json.loads(response["body"].read())

    print(model_response)

except Exception as e:
    print("Error:", e)