import boto3
import json
import base64

# class TextExtraction:
#     model_id = "us.amazon.nova-micro-v1:0"
#     region_name = "us-east-2"
#     service_name = "bedrock-runtime"
#     system = [{"text": "You are an AI assistant that provides only JSON formatted responses. Do not include any extra text, just return the JSON object."}]
#     inf_params = {"maxTokens": 150, "topP": 0.2, "topK": 20, "temperature": 0.5}

#     def __init__(self, s3_bucket, s3_key):
#         self.s3_key = s3_key
#         self.s3_bucket = s3_bucket

#         self.s3_client = boto3.client("s3")

#         self.bedrock_runtime = boto3.client(
#             service_name = TextExtraction.service_name,
#             region_name = TextExtraction.region_name
#         )

#     def get_s3_file_base64(self):
#         response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=self.s3_key)
#         return base64.b64encode(response["Body"].read()).decode("utf-8")
    
#     @staticmethod
#     def normalize_json_keys(json_data):
#         normalized_data = {}
#         for key, value in json_data.items():
#             normalized_key = key.replace(' ', '')
#             normalized_data[normalized_key] = value
#         return normalized_data
    
#     def nova_parser(self):
#         if self.s3_key.lower().endswith((".jpg", ".jpeg", ".png")):
#             content_type = "image/jpeg"
#         elif self.s3_key.lower().endswith(".pdf"):
#             content_type = "application/pdf"
#         else:
#             raise ValueError(f"Unsupported file format. Use PDF, JPG, or PNG.")
        
#         base64_data = self.get_s3_file_base64()

#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": content_type, "data": base64_data, "toolUse": None},
#                     {"type": "text", "text": "Provide only the following information in JSON format: Summary, Title, and Document Type.", "toolUse": None}
#                 ],
#             }
#         ]

#         payload = json.dumps(
#             {
#                 "schemaVersion": "messages-v1",
#                 "messages": messages,
#                 "system": TextExtraction.system,
#                 "inferenceConfig": TextExtraction.inf_params,
#             }
#         )

#         try: 
#             response = self.bedrock_runtime.invoke_model(
#                 modelId = TextExtraction.model_id,
#                 body = payload
#             )

#             model_response = json.loads(response["body"].read())
#             outputs = model_response.get("output").get("message").get("content")[0].get("text")

#             parsed_json = json.loads(outputs)
#             parsed_json = self.normalize_json_keys(parsed_json)
#             return parsed_json
        
#         except Exception as e:
#             print("Error:", e)
#             return "Error retrieving summary", "Error retrieving title"


# import base64
# import boto3
# import json

# class TextExtraction:
#     model_id = "us.amazon.nova-lite-v1:0"
#     region_name = "us-east-2"
#     service_name = "bedrock-runtime"

#     system = [
#         {"text": "You are an AI assistant that provides only JSON formatted responses. Do not include any extra text, just return the JSON object."}
#     ]
#     inf_params = {"maxTokens": 300, "topP": 0.1, "topK": 20, "temperature": 0.3}

#     def __init__(self, s3_bucket, s3_key):
#         self.s3_bucket = s3_bucket
#         self.s3_key = s3_key

#         self.s3_client = boto3.client("s3")
#         self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=TextExtraction.region_name)

#     def get_s3_file_base64(self):
#         """Retrieve the file from S3 and encode it as base64."""
#         response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=self.s3_key)
#         binary_data = response["Body"].read()
#         return base64.b64encode(binary_data).decode("utf-8")

#     def nova_parser(self):
#         """Send the extracted file content to Amazon Bedrock Nova Lite for processing."""
#         if self.s3_key.lower().endswith((".jpg", ".jpeg")):
#             file_format = "jpeg"  
#         elif self.s3_key.lower().endswith(".png"):
#             file_format = "png"
#         elif self.s3_key.lower().endswith(".pdf"):
#             file_format = "pdf"
#         else:
#             raise ValueError("Unsupported file format. Use PDF, JPG, or PNG.")

#         base64_data = self.get_s3_file_base64()

#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "image": {
#                             "format": file_format,
#                             "source": {"bytes": base64_data}
#                         }
#                     },
#                     {
#                         "text": "Extract and return the following information in JSON format: Summary, Title, and Document Type."
#                     }
#                 ],
#             }
#         ]

#         payload = json.dumps(
#             {
#                 "schemaVersion": "messages-v1",
#                 "messages": messages,
#                 "system": TextExtraction.system,
#                 "inferenceConfig": TextExtraction.inf_params,
#             }
#         )

#         try:
#             response = self.bedrock_runtime.invoke_model(
#                 modelId=TextExtraction.model_id,
#                 body=payload
#             )

#             model_response = json.loads(response["body"].read())
#             return model_response

#         except Exception as e:
#             print("Error:", e)
#             return {"Summary": "Error retrieving summary", "Title": "Error retrieving title"}


import base64
import boto3
import json
# import pdfplumber 
import io

class TextExtraction:
    model_id = "us.amazon.nova-lite-v1:0"
    region_name = "us-east-2"
    service_name = "bedrock-runtime"

    system = [
        {"text": "You are an AI assistant that provides only JSON formatted responses. Do not include any extra text, just return the JSON object."}
    ]
    inf_params = {"maxTokens": 300, "topP": 0.1, "topK": 20, "temperature": 0.3}

    def __init__(self, s3_bucket, s3_key):
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key

        self.s3_client = boto3.client("s3")
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=TextExtraction.region_name)

    def get_s3_file(self):
        """Retrieve the file from S3."""
        response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=self.s3_key)
        return response["Body"].read()

    # def extract_pdf_text(self, pdf_data):
    #     """Extract text from a PDF file."""
    #     text = ""
    #     with pdfplumber.open(pdf_data) as pdf:
    #         for page in pdf.pages:
    #             text += page.extract_text() + "\n"
    #     return text.strip()
    
    # def extract_pdf_text(self, pdf_data):
    #     """Extract text from a PDF file."""
    #     with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
    #         text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    #     return text

    def nova_parser(self):
        """Send image or extracted text to Amazon Bedrock Nova Lite."""
        if self.s3_key.lower().endswith((".jpg", ".jpeg", ".png")):
            file_format = "jpeg" if self.s3_key.lower().endswith((".jpg", ".jpeg")) else "png"
            base64_data = base64.b64encode(self.get_s3_file()).decode("utf-8")

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
                            "text": "Extract and return the following information in JSON format: Summary, Title, and Document Type."
                        }
                    ],
                }
            ]

        elif self.s3_key.lower().endswith(".pdf"):
            pdf_data = self.get_s3_file()
            extracted_text = self.extract_pdf_text(pdf_data)
            print(extracted_text)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": f"Extract and return the following information in JSON format: Summary, Title, and Document Type.\n\nDocument Content:\n{extracted_text}"
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
            response = self.bedrock_runtime.invoke_model(
                modelId=TextExtraction.model_id,
                body=payload
            )

            model_response = json.loads(response["body"].read())
            return model_response

        except Exception as e:
            print("Error:", e)
            return {"Summary": "Error retrieving summary", "Title": "Error retrieving title"}



s3_bucket = "billiaitest"
s3_key = "image.png" 
# s3_key = "DD214-Example_Redacted_0.pdf"
# s3_key = "diploma.jpg"

result = TextExtraction(s3_bucket, s3_key)
print(result.nova_parser())



        



    