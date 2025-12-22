# Ollama API Service

A secure, production-ready API gateway for Ollama with authentication, authorization, rate limiting, and comprehensive monitoring.

## Features

✅ **Authentication & Authorization**
- API key-based authentication
- Role-based access control (Admin/User)
- Secure key generation and management

✅ **Rate Limiting**
- Configurable rate limits per API key
- Per-hour request tracking
- Automatic rate limit enforcement

✅ **Monitoring & Analytics**
- Request tracking and logging
- Usage statistics per API key
- Response time monitoring
- Model usage analytics

✅ **API Documentation**
- Interactive Swagger UI documentation
- ReDoc alternative documentation
- Complete API reference

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd ollama-api-service

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit the `.env` file and configure your settings:

```env
# Change this to a secure random string in production
SECRET_KEY=your-super-secret-key-change-this-in-production

# Set your Linux VM IP address where Ollama is running
OLLAMA_BASE_URL=http://your-linux-vm-ip:11434

# Server configuration
HOST=0.0.0.0
PORT=8000
```

### 3. Run the Service

```bash
# Development mode
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The service will start at: `http://localhost:8000`

## API Documentation

Once the service is running, access the interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Demo API Keys

The service includes demo API keys for testing:

### Admin Key
```
API Key: 
Role: admin
Rate Limit: 1000 requests/hour
```

### User Key
```
API Key: 
Role: user
Rate Limit: 100 requests/hour
```

## API Usage Examples

### Authentication

All requests require an API key in the Authorization header:

```bash
Authorization: Bearer <your-api-key>
```

### 1. List Available Models

```bash
curl -X GET "http://localhost:8000/api/models" \
  -H "Authorization: Bearer demo-user-key-67890"
```

### 2. Generate Text

```bash
curl -X POST "http://localhost:8000/api/generate" \
  -H "Authorization: Bearer demo-user-key-67890" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "prompt": "Explain what is machine learning in simple terms",
    "temperature": 0.7,
    "stream": false
  }'
```

### 3. Chat Completion

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Authorization: Bearer demo-user-key-67890" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ],
    "temperature": 0.7,
    "stream": false
  }'
```

### 4. Get Usage Statistics

```bash
curl -X GET "http://localhost:8000/api/stats" \
  -H "Authorization: Bearer demo-user-key-67890"
```

### 5. Create New API Key (Admin Only)

```bash
curl -X POST "http://localhost:8000/api/keys" \
  -H "Authorization: Bearer demo-admin-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production API Key",
    "role": "user",
    "rate_limit": 500
  }'
```

### 6. List All API Keys (Admin Only)

```bash
curl -X GET "http://localhost:8000/api/keys" \
  -H "Authorization: Bearer demo-admin-key-12345"
```

### 7. Get All Usage Statistics (Admin Only)

```bash
curl -X GET "http://localhost:8000/api/admin/stats" \
  -H "Authorization: Bearer demo-admin-key-12345"
```

## Python Client Example

```python
import requests

API_BASE_URL = "http://localhost:8000"
API_KEY = "demo-user-key-67890"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# List models
response = requests.get(f"{API_BASE_URL}/api/models", headers=headers)
print("Available models:", response.json())

# Generate text
generate_payload = {
    "model": "llama2",
    "prompt": "Write a haiku about programming",
    "temperature": 0.8,
    "stream": False
}

response = requests.post(
    f"{API_BASE_URL}/api/generate",
    headers=headers,
    json=generate_payload
)
print("Generated text:", response.json())

# Chat completion
chat_payload = {
    "model": "llama2",
    "messages": [
        {"role": "user", "content": "Hello! How are you?"}
    ],
    "stream": False
}

response = requests.post(
    f"{API_BASE_URL}/api/chat",
    headers=headers,
    json=chat_payload
)
print("Chat response:", response.json())

# Get usage stats
response = requests.get(f"{API_BASE_URL}/api/stats", headers=headers)
print("Usage statistics:", response.json())
```

## API Endpoints Reference

### General Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Swagger UI documentation
- `GET /redoc` - ReDoc documentation

### API Key Management (Admin only)
- `POST /api/keys` - Create new API key
- `GET /api/keys` - List all API keys
- `DELETE /api/keys/{key_preview}` - Revoke API key

### Ollama Operations
- `GET /api/models` - List available models
- `POST /api/generate` - Generate text
- `POST /api/chat` - Chat completion

### Monitoring
- `GET /api/stats` - Get your usage statistics
- `GET /api/admin/stats` - Get all statistics (admin only)

## Rate Limiting

Each API key has a configurable rate limit (requests per hour):
- Default user rate limit: 100 requests/hour
- Default admin rate limit: 1000 requests/hour
- Custom limits can be set when creating API keys

When rate limit is exceeded, the API returns:
```json
{
  "detail": "Rate limit exceeded"
}
```
Status code: 429 (Too Many Requests)

## Security Best Practices

1. **Change the SECRET_KEY**: Always use a strong, randomly generated secret key in production
2. **Use HTTPS**: Deploy behind a reverse proxy with SSL/TLS
3. **Rotate API Keys**: Regularly rotate API keys
4. **Monitor Usage**: Review usage statistics for unusual patterns
5. **Firewall**: Restrict access to your Ollama VM
6. **Backup**: Store API keys securely (use a database in production)

## Production Deployment

### Using Docker (Recommended)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t ollama-api-service .
docker run -p 8000:8000 --env-file .env ollama-api-service
```

### Using systemd (Linux)

Create `/etc/systemd/system/ollama-api.service`:

```ini
[Unit]
Description=Ollama API Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/ollama-api-service
Environment="PATH=/opt/ollama-api-service/venv/bin"
ExecStart=/opt/ollama-api-service/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable ollama-api.service
sudo systemctl start ollama-api.service
```

## Monitoring and Logging

The service logs all activities including:
- API key authentication attempts
- Request processing
- Errors and exceptions
- Rate limit violations

Logs are written to stdout and can be redirected to a file or logging service.

## Troubleshooting

### Can't connect to Ollama backend

1. Check if Ollama is running on your Linux VM:
   ```bash
   curl http://your-vm-ip:11434/api/tags
   ```

2. Ensure Ollama is listening on all interfaces:
   ```bash
   # On your Linux VM
   OLLAMA_HOST=0.0.0.0:11434 ollama serve
   ```

3. Check firewall rules on your Linux VM

### Rate limit errors

Check your current usage:
```bash
curl -X GET "http://localhost:8000/api/stats" \
  -H "Authorization: Bearer your-api-key"
```

### Authentication errors

Verify your API key is correct and hasn't been revoked.

## License

MIT License - feel free to use this in your projects!

## Support

For issues and questions, please check the API documentation at `/docs` or review the logs for error messages.
