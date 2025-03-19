# file_recognition

# Workflow
The main file to run is file_recognition.py, the others are for testing.
The s3 bucket and s3 key is defined first.
The first function that is called is textract_parser, which takes in the s3 bucket and s3 key as paramters.
textract_parser analyzes the document using Amazon Textract and extracts: full body text and query results.
Based on the full body text, nova_micro_parser is called using the text as the parameter.
It uses Amazon Nova Micro to generate the summary, title, and document type based on the text.
The results are returned in json format to textract_parser.
Lastly, textract_parser appends all the data to the results in json format and returns.

# Activate the virtual environment
source .venv/Scripts/activate

# Dependencies
pillow
boto3
textract-trp

# s3 Keys
image.png
diplomatofboardcert.jpg
Phlebotomist-resume-1.png
lt11c.pdf
DD214-Example_Redacted_0.pdf
medicarcert.png
diploma.jpg

# Commands to find models
aws bedrock list-foundation-models --region us-east-2
aws bedrock list-inference-profiles --region us-east-2