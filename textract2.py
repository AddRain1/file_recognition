from PIL import Image, ImageEnhance, ImageFilter
import boto3
import json

def preprocess_image(image_path):
    image = Image.open(image_path)
    # Convert to grayscale
    image = image.convert('L')
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    # Apply a filter to sharpen the image
    image = image.filter(ImageFilter.SHARPEN)
    # Save the preprocessed image
    preprocessed_image_path = "preprocessed_image.jpg"
    image.save(preprocessed_image_path)
    return preprocessed_image_path

def textract_parser(s3_bucket, s3_key, queries):
    textract = boto3.client('textract', region_name='us-east-2')

    # call Textract with queries
    response = textract.analyze_document(
        Document={'S3Object': {'Bucket': s3_bucket, 'Name': s3_key}},
        FeatureTypes=["QUERIES"],
        QueriesConfig={"Queries": queries}
    )

    query_results = []
    confidence = []
    query_texts = [query["Text"] for query in queries]  # store questions in order

    # look for block with QUERY_RESULT type and store the answer
    for block in response.get('Blocks', []):
        if block.get('BlockType') == 'QUERY_RESULT':

            # if confidence isn't high enough, leave answer blank
            conf_level = block.get('Confidence', 0)
            res_text = block.get('Text', "")
            if conf_level > 90:
                query_results.append(res_text)
            else:
                query_results.append("")
            confidence.append(conf_level)

    # merge answers into the original query JSON format
    for i in range(len(queries)):
        queries[i]["Answer"] = query_results[i] if i < len(query_results) else ""
        queries[i]["Confidence"] = confidence[i] if i < len(confidence) else 0

    return(json.dumps(queries, indent=4))

# Preprocess the image before uploading to S3
preprocessed_image_path = preprocess_image("diploma.jpg")

# Upload the preprocessed image to S3
s3 = boto3.client('s3')
s3_bucket = "billiaitest"
s3_key = "preprocessed_diploma.jpg"
s3.upload_file(preprocessed_image_path, s3_bucket, s3_key)

queries = [
        {"Text": "What is the document type?"},
        {"Text": "What is the name of the document?"},
        {"Text": "What is the expiration date?"},
        {"Text": "What state is the document from?"},
        {"Text": "Are there any important comments?"}
]

result = textract_parser(s3_bucket, s3_key, queries)
print(result)