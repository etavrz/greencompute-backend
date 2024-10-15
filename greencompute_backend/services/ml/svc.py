import pickle

from loguru import logger

from greencompute_backend.config import AWS_S3_BUCKET


class PredictionService:
    def __init__(self, model_name: str, s3_client):
        self.s3_client = s3_client
        self.model = self._download_model(model_name)

    def _download_model(self, model_name: str):
        response = self.s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=model_name)
        model_bytes = response["Body"].read()
        logger.debug("✅ Successfully downloaded model from S3")
        return pickle.loads(model_bytes)

    def predict(self, data):
        return self.model.predict(data)
