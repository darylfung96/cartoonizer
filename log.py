import boto3
import uuid
import time
import os

from PIL import Image
from io import BytesIO


MAX_PIXELS = 2048

AWS_BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME", "")
AWS_INFERENCE_LOG_TABLE = os.environ.get("AWS_INFERENCE_LOG_TABLE", "")
AWS_FEEDBACK_LOG_TABLE = os.environ.get("AWS_FEEDBACK_LOG_TABLE", "")


AWS_REGION = os.environ.get("AWS_REGION", "")
AWS_ACCESS_ID = os.environ.get("AWS_ACCESS_ID", "")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", "")


aws_cfg = {
    "aws_access_key_id": AWS_ACCESS_ID,
    "aws_secret_access_key": AWS_ACCESS_KEY,
    "region_name": AWS_REGION,
}

s3_client = boto3.client("s3", **aws_cfg)
dynamodb = boto3.resource("dynamodb", **aws_cfg)

inference_log = dynamodb.Table(AWS_INFERENCE_LOG_TABLE)
feedback_log = dynamodb.Table(AWS_FEEDBACK_LOG_TABLE)


def get_metadata():
    return {
        "_id": uuid.uuid4().hex,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }


def insert_log(table_type: str, data: dict):
    assert table_type in ["inference", "feedback"], "Invalid table type"
    table = inference_log if table_type == "inference" else feedback_log
    metadata = get_metadata()
    response = table.put_item(
        Item={
            **data,
            **metadata,
        }
    )
    return response, metadata["_id"]


# Example usage:
# insert_log("inference", {"data": "test"})
# insert_log("feedback", {"data": "test"})


def get_image_obj(image: Image) -> BytesIO:
    image.thumbnail((MAX_PIXELS, MAX_PIXELS))
    image_obj = BytesIO()
    image.save(image_obj, format="WEBP")
    image_obj.seek(0)
    return image_obj


def log_image(image: Image) -> str:
    metadata = get_metadata()
    image_obj = get_image_obj(image)
    s3_key = f"images/{metadata['_id']}.webp"
    s3_client.upload_fileobj(image_obj, AWS_BUCKET_NAME, s3_key)
    return metadata["_id"]


# Example usage:
# image = Image.open("examples/doge.jpg")
# log_image(image)
