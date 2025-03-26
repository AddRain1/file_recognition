import boto3
import time

client = boto3.client("bedrock-runtime", region_name="us-east-2")

def textract_parser(s3_bucket, s3_key):
    textract = boto3.client("textract", region_name="us-east-2")

    response = textract.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": s3_bucket, "Name": s3_key}}
    )

    job_id = response["JobId"]
    print(f"Started job with ID: {job_id}")

    while True:
        response = textract.get_document_text_detection(JobId=job_id)
        status = response["JobStatus"]
        if status in ["SUCCEEDED", "FAILED"]:
            break
        time.sleep(5)

    if status == "SUCCEEDED":
        for block in response["Blocks"]:
            if block["BlockType"] == "LINE":
                print(block["Text"])



s3_bucket = "billiaitest"
s3_key = "DD214-Example_Redacted_0.pdf"
s3_key = "lt11c.pdf"

result = textract_parser(s3_bucket, s3_key)