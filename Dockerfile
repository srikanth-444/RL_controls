FROM python:3.12.6-slim

WORKDIR /app

# Install gRPC and other dependencies
COPY requirements.txt .
RUN ls -l requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and proto files
COPY data/ ./data
COPY envirnoment/ ./envirnoment
COPY models/ ./models
COPY policy/ ./policy
COPY generated/ ./generated
COPY rollout.proto .
COPY runrollout.py .
COPY tinyphysics.py .


# If you want to compile .proto files in Docker (optional)
RUN mkdir -p generated
RUN python -m grpc_tools.protoc -I. --python_out=generated --grpc_python_out=generated rollout.proto

EXPOSE 50051

EXPOSE 8080

CMD ["sh", "-c", "python3 -m http.server 8080 & python runrollout.py"]
