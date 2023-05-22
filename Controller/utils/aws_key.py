import boto3

session = boto3.Session()
credentials = session.get_credentials()

CREDENTIALS = {
    "aws_access_key_id": credentials.access_key,
    "aws_secret_access_key": credentials.secret_key
}
