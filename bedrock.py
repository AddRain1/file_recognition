import boto3
import json

# Initialize Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-2")

# Define input text
prompt = "Summarize this: It could not have been ten seconds, and yet it seemed a long time that their hands were clasped together.  He had time to learn every detail of her hand.  He explored the long fingers, the shapely nails, the work-hardened palm with its row of callouses, the smooth flesh under the wrist.  Merely from feeling it he would have known it by sight.  In the same instant it occurred to him that he did not know what colour the girl's eyes were.  They were probably brown, but people with dark hair sometimes had blue eyes.  To turn his head and look at her would have been inconceivable folly.  With hands locked together, invisible among the press of bodies, they stared steadily in front of them, and instead of the eyes of the girl, the eyes of the aged prisoner gazed mournfully at Winston out of nests of hair."

# inference_profile_id = "us.anthropic.claude-3-haiku-20240307-v1:0"
modelID = "meta.llama3-3-70b-instruct-v1:0"

# Set up the payload
body = json.dumps({
    "prompt": prompt,
    "max_gen_len": 200,
    "temperature": 0.5,
    "top_p": 1.0,
})

accept="application/json"
contentType = "application/json"

# Invoke Claude 3 Haiku (200k version)
response = bedrock.invoke_model(
    modelId = modelID,
    body = body,
    accept = accept,
    contentType = contentType
)

print(json.loads(response.get('body').read()))
