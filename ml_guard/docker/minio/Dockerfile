FROM quay.io/minio/minio:latest

# MinIO runs on ports 9000 (API) and 9001 (Console)
EXPOSE 9000
EXPOSE 9001

# The command will be provided by Render's 'Docker Command' field or defaults to server /data
CMD ["minio", "server", "/data", "--console-address", ":9001"]
