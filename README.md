# file_recognition

# Workflow

The main file to run is file_extraction.py.

An instance of the TextExtraction class is initialized with the s3 bucket and s3 key.
If the file is an image (.png, jpeg, .jpg), the image data is encoded as base64 and
data is extracted using Amazon Textract and Amazon Nova Lite. On the other hand,
if the fils is a PDF, the data is extracted using Amazon Textract and Amazon Nova Micro.
Text can be extracted from multipage documents.

## Output Format

```
{

    "Summary": "",
    "Title": "",
    "Document Type": "",
    "Expiration Date": {
        "value": "",
        "confidence": 0.00
    },
    "State": {
        "value": "",
        "confidence": 0.00
    }
}
```

# Activate the virtual environment

source .venv/Scripts/activate

# Dependencies

boto3
Amazon textract
Amazon nova-lite
Amazon nova-micro

# s3 Keys

image.png
lt11c.pdf
DD214-Example_Redacted_0.pdf
diploma.jpg
fw9.pdf

# Commands to find models

aws bedrock list-foundation-models --region us-east-2
aws bedrock list-inference-profiles --region us-east-2
