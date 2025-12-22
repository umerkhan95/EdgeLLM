# Ollama API Service - Complete API Documentation

## Table of Contents
1. [Authentication](#authentication)
2. [Rate Limiting](#rate-limiting)
3. [Error Handling](#error-handling)
4. [Endpoints](#endpoints)
   - [General](#general-endpoints)
   - [API Key Management](#api-key-management)
   - [Ollama Operations](#ollama-operations)
   - [Monitoring](#monitoring)
5. [Request/Response Examples](#request-response-examples)
6. [SDK Examples](#sdk-examples)

---

## Authentication

All API requests require authentication using an API key in the `Authorization` header:

```
Authorization: Bearer <your-api-key>
```

### Getting an API Key

Contact your administrator to create an API key for you. For testing, you can set demo keys in your `.env` file:

**Demo Keys Configuration:**
```bash
# In .env file:
DEMO_ADMIN_KEY=<your-generated-admin-key>
DEMO_USER_KEY=<your-generated-user-key>

# Generate secure keys with:
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Roles

- **admin**: Full access including API key management and system statistics
- **user**: Access to Ollama operations and personal statistics

---

## Rate Limiting

Each API key has a configurable rate limit (requests per hour). When exceeded, you'll receive:

```json
{
  "detail": "Rate limit exceeded"
}
```

**HTTP Status:** `429 Too Many Requests`

Check your current usage with `GET /api/stats`

---

## Error Handling

### Standard Error Response

```json
{
  "detail": "Error message description"
}
```

### HTTP Status Codes

- `200` - Success
- `401` - Unauthorized (invalid or missing API key)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found
- `429` - Too Many Requests (rate limit exceeded)
- `500` - Internal Server Error

---

## Endpoints

### General Endpoints

#### `GET /`
Get API information

**Authentication:** Not required

**Response:**
```json
{
  "service": "Ollama API Service",
  "version": "1.0.0",
  "status": "running",
  "documentation": "/docs",
  "health": "/health"
}
```

---

#### `GET /health`
Health check endpoint

**Authentication:** Not required

**Response:**
```json
{
  "status": "healthy",
  "ollama_backend": "healthy",
  "timestamp": "2025-12-22T10:30:00.000000"
}
```

**Backend Status Values:**
- `healthy` - Ollama is responding normally
- `unhealthy` - Ollama returned an error
- `unreachable` - Cannot connect to Ollama

---

### API Key Management

#### `POST /api/keys`
Create a new API key

**Authentication:** Required (Admin only)

**Request Body:**
```json
{
  "name": "Production API Key",
  "role": "user",
  "rate_limit": 500
}
```

**Parameters:**
- `name` (required): Description for the API key
- `role` (optional): "admin" or "user" (default: "user")
- `rate_limit` (optional): Requests per hour (default: 100)

**Response:**
```json
{
  "api_key": "ollama-abc123def456...",
  "name": "Production API Key",
  "role": "user",
  "rate_limit": 500,
  "created_at": "2025-12-22T10:30:00.000000"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/keys" \
  -H "Authorization: Bearer <your-admin-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production API Key",
    "role": "user",
    "rate_limit": 500
  }'
```

---

#### `GET /api/keys`
List all API keys

**Authentication:** Required (Admin only)

**Response:**
```json
{
  "api_keys": [
    {
      "key_preview": "<admin-key-preview>...",
      "name": "Demo Admin Key",
      "role": "admin",
      "rate_limit": 1000,
      "created_at": "2025-12-22T10:00:00.000000"
    }
  ],
  "total": 1
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/keys" \
  -H "Authorization: Bearer <your-admin-api-key>"
```

---

#### `DELETE /api/keys/{key_preview}`
Revoke an API key

**Authentication:** Required (Admin only)

**Path Parameters:**
- `key_preview`: First 20 characters of the API key

**Response:**
```json
{
  "message": "API key revoked successfully"
}
```

**cURL Example:**
```bash
curl -X DELETE "http://localhost:8000/api/keys/<your-user-api-key>" \
  -H "Authorization: Bearer <your-admin-api-key>"
```

---

### Ollama Operations

#### `GET /api/models`
List available Ollama models

**Authentication:** Required

**Response:**
```json
{
  "models": [
    {
      "name": "llama2",
      "modified_at": "2025-12-20T15:30:00Z",
      "size": 3826793677,
      "digest": "sha256:abc123..."
    },
    {
      "name": "mistral",
      "modified_at": "2025-12-21T09:15:00Z",
      "size": 4109865159,
      "digest": "sha256:def456..."
    }
  ]
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/models" \
  -H "Authorization: Bearer <your-user-api-key>"
```

---

#### `POST /api/generate`
Generate text using an Ollama model

**Authentication:** Required

**Request Body:**
```json
{
  "model": "llama2",
  "prompt": "Explain quantum computing in simple terms",
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 500
}
```

**Parameters:**
- `model` (required): Model name (e.g., "llama2", "mistral")
- `prompt` (required): Text prompt for generation
- `stream` (optional): Stream response (default: false)
- `temperature` (optional): Sampling temperature 0.0-1.0 (default: 0.7)
- `max_tokens` (optional): Maximum tokens to generate

**Response:**
```json
{
  "model": "llama2",
  "created_at": "2025-12-22T10:30:00.000000Z",
  "response": "Quantum computing is a revolutionary type of computing...",
  "done": true,
  "total_duration": 5240250000,
  "load_duration": 1234567,
  "prompt_eval_count": 26,
  "eval_count": 298
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/generate" \
  -H "Authorization: Bearer <your-user-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "prompt": "Write a haiku about programming",
    "temperature": 0.8,
    "stream": false
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/generate",
    headers={
        "Authorization": "Bearer <your-user-api-key>",
        "Content-Type": "application/json"
    },
    json={
        "model": "llama2",
        "prompt": "Explain machine learning",
        "temperature": 0.7,
        "max_tokens": 300
    }
)

print(response.json()["response"])
```

---

#### `POST /api/chat`
Chat with an Ollama model

**Authentication:** Required

**Request Body:**
```json
{
  "model": "llama2",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful coding assistant."
    },
    {
      "role": "user",
      "content": "How do I reverse a string in Python?"
    }
  ],
  "stream": false,
  "temperature": 0.7
}
```

**Parameters:**
- `model` (required): Model name
- `messages` (required): Array of message objects
  - `role`: "system", "user", or "assistant"
  - `content`: Message text
- `stream` (optional): Stream response (default: false)
- `temperature` (optional): Sampling temperature 0.0-1.0 (default: 0.7)

**Response:**
```json
{
  "model": "llama2",
  "created_at": "2025-12-22T10:30:00.000000Z",
  "message": {
    "role": "assistant",
    "content": "To reverse a string in Python, you can use slicing: `reversed_string = my_string[::-1]`"
  },
  "done": true,
  "total_duration": 3240250000
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Authorization: Bearer <your-user-api-key>" \
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
    "stream": false
  }'
```

**Python Example:**
```python
import requests

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain recursion with an example"}
]

response = requests.post(
    "http://localhost:8000/api/chat",
    headers={
        "Authorization": "Bearer <your-user-api-key>",
        "Content-Type": "application/json"
    },
    json={
        "model": "llama2",
        "messages": messages,
        "temperature": 0.7
    }
)

print(response.json()["message"]["content"])
```

---

### Monitoring

#### `GET /api/stats`
Get usage statistics for your API key

**Authentication:** Required

**Response:**
```json
{
  "total_requests": 42,
  "requests_by_model": {
    "llama2": 30,
    "mistral": 12
  },
  "requests_by_endpoint": {
    "/api/generate": 25,
    "/api/chat": 15,
    "/api/models": 2
  },
  "average_response_time": 2.456,
  "last_24h_requests": 38
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/stats" \
  -H "Authorization: Bearer <your-user-api-key>"
```

---

#### `GET /api/admin/stats`
Get usage statistics for all API keys

**Authentication:** Required (Admin only)

**Response:**
```json
{
  "statistics": {
    "<admin-key-preview>...": {
      "key_name": "Demo Admin Key",
      "role": "admin",
      "total_requests": 15,
      "last_request": "2025-12-22T10:25:00.000000"
    },
    "<your-user-api-key>...": {
      "key_name": "Demo User Key",
      "role": "user",
      "total_requests": 42,
      "last_request": "2025-12-22T10:30:00.000000"
    }
  }
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/admin/stats" \
  -H "Authorization: Bearer <your-admin-api-key>"
```

---

## SDK Examples

### JavaScript/Node.js

```javascript
const axios = require('axios');

const API_BASE_URL = 'http://localhost:8000';
const API_KEY = '<your-user-api-key>';

const headers = {
  'Authorization': `Bearer ${API_KEY}`,
  'Content-Type': 'application/json'
};

// List models
async function listModels() {
  const response = await axios.get(`${API_BASE_URL}/api/models`, { headers });
  return response.data;
}

// Generate text
async function generate(model, prompt) {
  const response = await axios.post(
    `${API_BASE_URL}/api/generate`,
    {
      model: model,
      prompt: prompt,
      temperature: 0.7
    },
    { headers }
  );
  return response.data;
}

// Chat
async function chat(model, messages) {
  const response = await axios.post(
    `${API_BASE_URL}/api/chat`,
    {
      model: model,
      messages: messages,
      temperature: 0.7
    },
    { headers }
  );
  return response.data;
}

// Usage
(async () => {
  const models = await listModels();
  console.log('Available models:', models);
  
  const result = await generate('llama2', 'Hello, world!');
  console.log('Generated:', result.response);
  
  const chatResult = await chat('llama2', [
    { role: 'user', content: 'Hi there!' }
  ]);
  console.log('Chat:', chatResult.message.content);
})();
```

### cURL Complete Workflow

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. List models
curl -X GET "http://localhost:8000/api/models" \
  -H "Authorization: Bearer <your-user-api-key>"

# 3. Generate text
curl -X POST "http://localhost:8000/api/generate" \
  -H "Authorization: Bearer <your-user-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "prompt": "What is AI?",
    "temperature": 0.7
  }'

# 4. Chat completion
curl -X POST "http://localhost:8000/api/chat" \
  -H "Authorization: Bearer <your-user-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'

# 5. Get your stats
curl -X GET "http://localhost:8000/api/stats" \
  -H "Authorization: Bearer <your-user-api-key>"

# Admin: Create API key
curl -X POST "http://localhost:8000/api/keys" \
  -H "Authorization: Bearer <your-admin-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "New User Key",
    "role": "user",
    "rate_limit": 200
  }'
```

---

## Best Practices

1. **Store API keys securely**: Never commit API keys to version control
2. **Handle rate limits**: Implement exponential backoff when you receive 429 errors
3. **Monitor usage**: Regularly check your usage statistics
4. **Use appropriate temperature**: Lower (0.1-0.3) for factual responses, higher (0.7-1.0) for creative content
5. **Set max_tokens**: Prevent unexpectedly long responses by setting reasonable limits
6. **Error handling**: Always handle network errors and API errors gracefully

---

## Support

For interactive API documentation, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

For issues or questions, check the application logs or contact your administrator.
