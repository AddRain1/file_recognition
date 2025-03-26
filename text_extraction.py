import boto3
import base64
import json
import time
import re

document_types = [
    "Application Release",
    "Board Certification",
    "CDS Certificate",
    "Certificate or Letter Certifying Formal Post-Graduate Training",
    "Certificates of Completion (Med School, Internship etc)",
    "CLIA/COLA/CAP Certification",
    "CME Certification",
    "CPR Card",
    "DD214, Record of Military Service",
    "DEA",
    "DEA Waiver",
    "Diplomat of National Board of Medical Examiners Certificate",
    "Disclosure",
    "DPS",
    "ECFMG",
    "Federal Tort Claim Act Coverage",
    "Form A - Adverse and other actions",
    "Form B - Professional Liability Actions",
    "Form C - Liability Insurance",
    "Form D - Criminal Actions",
    "Form E - Medical Condition",
    "Form F - Chemical Substances or Alcohol Abuse",
    "Hospital Letter, Verification of Hospital credentialing or Alternative Pathways",
    "Immunization Certificate of Achievement",
    "Letter of Self Insurance/Explanation of No Insurance",
    "Resume",
    "Schedule C - Regulation Acknowledgement",
    "Schedule B - Professional Liability Claims Information Form for Georgia State",
    "Section D - Attestation Questions",
    "State Authorization",
    "State License",
    "State Release",
    "Supervisory/Collaboration Agreement",
    "TB Skin Test",
    "Therapeutic/Diagnostic Pharmaceutical Agents License",
    "W-9",
    "Written Protocol",
    "Driver's License",
    "ABA Certification",
    "AHCA background screening",
    "Master's degree",
    "IRS letter",
    "Voided check",
    "Collaborative agreement (for nurse practitioners only)",
    "Bachelor's degree",
    "Diploma",
    "Medicare approval letter",
    "Medicaid approval letter",
    "OIG Verification",
    "NPI verification",
    "Social Security Number",
    "FL AHCA State License (If applicable)",
    "Credentialed Attestation",
    "Other Documents",
    "Active Credentialling Proof"
]


json_format = {
    
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

class TextExtraction:
    model_lite_id = "us.amazon.nova-lite-v1:0"
    model_micro_id = "us.amazon.nova-micro-v1:0"
    region_name = "us-east-2"
    service_name = "bedrock-runtime"

    system = [
        {"text": "You are an AI assistant that provides only JSON formatted responses. Do not include any extra text, just return the JSON object."}
    ]
    inf_params = {"maxTokens": 300, "topP": 0.2, "topK": 20, "temperature": 0.5}

    def __init__(self, s3_bucket, s3_key):
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key

        self.s3_client = boto3.client("s3")
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=TextExtraction.region_name)

    def get_s3_file(self):
        response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=self.s3_key)
        return response["Body"].read()
    
    def textract_parser(self):
        textract = boto3.client("textract", region_name=TextExtraction.region_name)
        response = textract.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": self.s3_bucket, "Name": self.s3_key}}
        )

        job_id = response["JobId"]

        while True:
            response = textract.get_document_text_detection(JobId=job_id)
            status = response["JobStatus"]
            if status in ["SUCCEEDED", "FAILED"]:
                break
            time.sleep(30)

        extracted_texts = []
        if status == "SUCCEEDED":
            for block in response["Blocks"]:
                if block["BlockType"] == "LINE":
                    extracted_texts.append(block["Text"])

        return " ".join(extracted_texts)

    @staticmethod
    def clean_string(string):
        try:
            if not isinstance(string, str):
                raise ValueError("Input must be a string")

            string = string.strip()

            try:
                string = string.encode().decode("unicode_escape")
            except UnicodeDecodeError:
                pass 

            string = string.strip("`")

            string = re.sub(r"^json|json$", "", string, flags=re.IGNORECASE).strip()

            return json.loads(string)

        except json.JSONDecodeError as e:
            print(f"Error in clean_string: {e}")
            return string 
    
    def nova_parser(self):

        text_content = None

        if self.s3_key.lower().endswith((".jpg", ".jpeg", ".png")):
            file_format = "jpeg" if self.s3_key.lower().endswith((".jpg", ".jpeg")) else "png"
            base64_data = base64.b64encode(self.get_s3_file()).decode("utf-8")
            text_content = self.textract_parser()

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": file_format,
                                "source": {"bytes": base64_data}
                            }
                        },
                        {
                            "text": f"Extract and return the following information in JSON format: {json_format}. Provide confidence scores for Expiration Date and State. Choose a Document Type from this list: {document_types}."
                        }
                    ],
                }
            ]

        elif self.s3_key.lower().endswith(".pdf"):
            text_content = self.textract_parser()
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"text": text_content},
                        {
                            "text": f"Extract and return the following information in JSON format: {json_format}. Provide confidence scores for Expiration Date and State. Choose a Document Type from this list: {document_types}."
                        }
                    ],
                }
            ]

        else:
            raise ValueError("Unsupported file format. Use PDF, JPG, or PNG.")
        
        payload = json.dumps(
            {
                "schemaVersion": "messages-v1",
                "messages": messages,
                "system": TextExtraction.system,
                "inferenceConfig": TextExtraction.inf_params,
            }
        )

        try:
            if self.s3_key.lower().endswith((".jpg", ".jpeg", ".png")):
                bedrock_response = self.bedrock_runtime.invoke_model(
                    modelId=TextExtraction.model_lite_id,
                    body=payload
                )

                model_response = json.loads(bedrock_response["body"].read())


            elif self.s3_key.lower().endswith(".pdf"):

                bedrock_response = self.bedrock_runtime.invoke_model(
                    modelId=TextExtraction.model_micro_id,
                    body=payload
                )

                model_response = json.loads(bedrock_response["body"].read())

            else:
                model_response = {}
            
            structured_string =  model_response.get("output", {}).get("message", {}).get("content", {})[0].get("text", {}) if model_response else model_response

            cleaned_string = self.clean_string(structured_string)

            structured_dict = cleaned_string
            
            structured_dict["Full Text"] = text_content
            
            return json.dumps(structured_dict, indent=4)


        except Exception as e:
            return json.dumps({"Error": f"{e}"})



s3_bucket = "billiaitest"
s3_key = "image.png" 
s3_key = "DD214-Example_Redacted_0.pdf"
s3_key = "diploma.jpg"
s3_key = "lt11c.pdf"

result = TextExtraction(s3_bucket, s3_key)
print(result.nova_parser())