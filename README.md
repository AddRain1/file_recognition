# file_recognition

# Workflow

The main file to run is text_extraction.py.

The text_extraction.py file implements a TextExtraction class that processes documents stored in an S3 bucket using Amazon Textract and Bedrock models (Nova Lite and Nova Micro). It supports both image files (e.g., .jpg, .jpeg, .png) and PDFs, extracting text and structured information such as Summary, Title, Document Type, Expiration Date, and State. The class initializes with S3 bucket and key details, retrieves files from S3, and uses Textract for text detection. It includes helper methods like _start_textract_job, _wait_for_textract_job, and _get_textract_results for handling Textract jobs, with dynamic polling intervals for efficiency. The nova_parser method processes the extracted text using Bedrock models, formats the results into a structured JSON format, and includes logic to set Summary to Title if Summary is empty. Additional utilities like clean_string, format_date, and confidence_format handle data cleaning, date formatting, and confidence score normalization. The script also supports parallel processing of multiple files using concurrent.futures.ThreadPoolExecutor, with results printed in JSON format.

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

```
source .venv/Scripts/activate
```

# Dependencies

```
boto3
Amazon textract
Amazon nova-lite
Amazon nova-micro
```

# s3 Keys

```
image.png
lt11c.pdf
DD214-Example_Redacted_0.pdf
diploma.jpg
fw9.pdf
Form C - Liability Insurance.pdf
Social Security Number.jpg
CPR Card.png
Letter of Self Insurance.pdf
TB Skin Test.jpg
DEA Waiver.pdf
CME Certification.pdf
TherapeuticDiagnostic Pharmaceutical Agents License.pdf
```

# Commands to find models

```
aws bedrock list-foundation-models --region us-east-2
aws bedrock list-inference-profiles --region us-east-2
```
