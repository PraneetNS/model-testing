
import boto3
from botocore.config import Config
import os
from dotenv import load_dotenv

load_dotenv()

def diag():
    endpoint = os.getenv("MINIO_ENDPOINT", "https://minio1-uwny.onrender.com")
    access = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    bucket = os.getenv("MINIO_BUCKET", "mlguard-artifacts")
    
    print(f"Connecting to: {endpoint}")
    print(f"Access Key: {access}")
    print(f"Bucket: {bucket}")
    
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"})
    )
    
    try:
        print("Checking bucket...")
        s3.head_bucket(Bucket=bucket)
        print("✅ Bucket exists and is accessible.")
        
        print("Testing upload...")
        s3.put_object(Bucket=bucket, Key="diag_test.txt", Body=b"ML Guard connection test")
        print("✅ Upload successful!")
        
        print("Cleaning up...")
        s3.delete_object(Bucket=bucket, Key="diag_test.txt")
        print("✅ Cleanup successful!")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")

if __name__ == "__main__":
    diag()
