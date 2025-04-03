import boto3
import base64
import json
import time
from datetime import datetime as dt
import re
import concurrent.futures
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        "confidence": ""
    },
    "State": {
        "value": "",
        "confidence": ""
    }
}

class TextExtraction:
    model_lite_id = "us.amazon.nova-lite-v1:0"
    model_micro_id = "us.amazon.nova-micro-v1:0"
    region_name = "us-east-2"
    service_name = "bedrock-runtime"
    max_concurrent_jobs = 5

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
        try:
            textract = boto3.client("textract", region_name=TextExtraction.region_name)
            response = textract.start_document_text_detection(
                DocumentLocation={"S3Object": {"Bucket": self.s3_bucket, "Name": self.s3_key}}
            )

            job_id = response["JobId"]
            self._wait_for_textract_job(textract, job_id)
            return self._get_textract_results(textract, job_id)

        except boto3.exceptions.Boto3Error as e:
            logger.error(f"Error with Textract: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return ""

    def _wait_for_textract_job(self, textract, job_id):
        while True:
            response = textract.get_document_text_detection(JobId=job_id)
            status = response["JobStatus"]

            if status == "FAILED":
                logger.error("Textract job failed")
                raise RuntimeError("Textract job failed")
            elif status == "SUCCEEDED":
                logger.info("Textract job succeeded")
                break

            logger.info("Job in progress...")
            time.sleep(2)

    def _get_textract_results(self, textract, job_id):
        extracted_texts = []
        next_token = None

        while True:
            params = {"JobId": job_id}
            if next_token:
                params["NextToken"] = next_token

            response = textract.get_document_text_detection(**params)
            for block in response.get("Blocks", []):
                if block["BlockType"] == "LINE":
                    extracted_texts.append(block["Text"])

            next_token = response.get("NextToken")
            if not next_token:
                break

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

            structured_dict = self.clean_string(structured_string)
            
            structured_dict["Full Text"] = text_content

            for dict_key, dict_value in structured_dict.items():

                if dict_key.lower() == "expiration date":

                    if isinstance(dict_value, dict):
                    
                        value = dict_value.get("value", "")

                        if value.startswith("[") and value.endswith("]"):
                            value = value[1:-1]

                        if value:
                            value = self.format_date(value)
                        
                        dict_value["value"] = value

                elif dict_key.lower() == "state":

                    if isinstance(dict_value, dict):

                        value = dict_value.get("value", "")

                        if value.lower() in ["n/a", "na", "not applicable"]:
                            value = ""

                        dict_value["value"] = value


                if dict_key.lower() == "expiration date" or dict_key.lower() == "state":

                    if isinstance(dict_value, dict):
                        
                        confidence = dict_value.get("confidence", "")

                        if confidence:
                            confidence = self.confidence_format(confidence)

                            value = dict_value.get("value")

                            if not value:
                                confidence = ""
                        
                        dict_value["confidence"] = confidence
                        
                
            # return json.dumps(structured_dict, indent=4)
            return structured_dict


        except Exception as e:
            # return json.dumps({"Error": f"{e}"})
            return {"Error": f"{e}"}
        
    
    @staticmethod

    def format_date(date_str):
        if not date_str:
            return ""

        date_str = date_str.replace(".", "/")

        if re.fullmatch(r"0{2,4}[-/.]0{2}[-/.]0{2,4}", date_str):
            return ""

        if date_str.replace(" ", "").isalpha():
            return ""

        if re.fullmatch(r"\d{4}", date_str):
            return date_str

        month_year_formats = [
            "%B %Y", "%B/%Y", "%B-%Y",  # Full month name (March 2024)
            "%b %Y", "%b/%Y", "%b-%Y",  # Abbreviated month name (Mar 2024)
            "%m/%Y", "%Y/%m", "%Y-%m",  # Numeric Month-Year (03/2024)
        ]

        for fmt in month_year_formats:
            try:
                parsed_date = dt.strptime(date_str, fmt)
                return parsed_date.strftime("%m-%Y")
            except ValueError:
                pass

        full_date_formats = [
            "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",  # ISO formats (YYYY-MM-DD)
            "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y",  # US format (MM/DD/YYYY)
            "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",  # European format (DD/MM/YYYY)
            "%Y %m %d", "%d %m %Y", "%m %d %Y",  # Spaces instead of symbols
        ]

        for fmt in full_date_formats:
            try:
                parsed_date = dt.strptime(date_str, fmt)
                return parsed_date.strftime("%m-%d-%Y")
            except ValueError:
                pass

        return date_str

    @staticmethod
    def confidence_format(conf_str):

        if isinstance(conf_str, float) or isinstance(conf_str, int):
            return float(conf_str)
        
        if isinstance(conf_str, str):
            try:
                return float(conf_str)
            except ValueError:
                pass

        return conf_str







def parallel_processing(s3_bucket, s3_key):
    text_extraction = TextExtraction(s3_bucket, s3_key)
    result = text_extraction.nova_parser()
    return json.dumps({"file": s3_key, "result": result}, indent=4)


def process_files_in_parallel(s3_bucket, s3_keys):
    with concurrent.futures.ThreadPoolExecutor(max_workers=TextExtraction.max_concurrent_jobs) as executor:
        future_to_key = {executor.submit(parallel_processing, s3_bucket, key): key for key in s3_keys}
        for future in concurrent.futures.as_completed(future_to_key):
            try:
                result = future.result()
                print(result)
            except Exception as e:
                logger.error(f"Error processing file {future_to_key[future]}: {e}")


s3_bucket = "billiaitest"
s3_keys = ["TherapeuticDiagnostic Pharmaceutical Agents License.pdf", 
            "Form C - Liability Insurance.pdf",
            "Social Security Number.jpg",
            "CPR Card.png",
            "Letter of Self Insurance.pdf",
            "TB Skin Test.jpg",
            "DEA Waiver.pdf",
            "CME Certification.pdf",
            "TherapeuticDiagnostic Pharmaceutical Agents License.pdf" 
]

"""
Form C - Liability Insurance.pdf
Social Security Number.jpg
CPR Card.png
Letter of Self Insurance.pdf
TB Skin Test.jpg
DEA Waiver.pdf
CME Certification.pdf
TherapeuticDiagnostic Pharmaceutical Agents License.pdf
"""

process_files_in_parallel(s3_bucket, s3_keys)
