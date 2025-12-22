"""
Ollama API Service with Authentication, Authorization, and Monitoring
"""
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import jwt
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import logging
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# In-memory storage (use database in production)
API_KEYS: Dict[str, Dict[str, Any]] = {}
USAGE_STATS: Dict[str, List[Dict[str, Any]]] = {}

# Security
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("Starting Ollama API Service...")
    # Initialize demo API keys
    initialize_demo_keys()
    yield
    # Shutdown
    logger.info("Shutting down Ollama API Service...")

# Initialize FastAPI app
app = FastAPI(
    title="Ollama API Service",
    description="Secure API gateway for Ollama with authentication, authorization, and monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class User(BaseModel):
    username: str
    role: str = Field(default="user", description="User role: admin or user")

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class APIKeyCreate(BaseModel):
    name: str = Field(..., description="Name/description for this API key")
    role: str = Field(default="user", description="Role: admin or user")
    rate_limit: int = Field(default=100, description="Requests per hour")

class APIKeyResponse(BaseModel):
    api_key: str
    name: str
    role: str
    rate_limit: int
    created_at: str

class GenerateRequest(BaseModel):
    model: str = Field(..., description="Model name (e.g., llama2, mistral)")
    prompt: str = Field(..., description="The prompt to generate from")
    stream: bool = Field(default=False, description="Stream the response")
    temperature: Optional[float] = Field(default=0.7, description="Temperature for generation")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    model: str = Field(..., description="Model name")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    stream: bool = Field(default=False, description="Stream the response")
    temperature: Optional[float] = Field(default=0.7, description="Temperature for generation")

class UsageStats(BaseModel):
    total_requests: int
    requests_by_model: Dict[str, int]
    requests_by_endpoint: Dict[str, int]
    average_response_time: float
    last_24h_requests: int

# Helper Functions
def initialize_demo_keys():
    """Initialize demo API keys for testing"""
    demo_keys = [
        {
            "key": "demo-admin-key-12345",
            "name": "Demo Admin Key",
            "role": "admin",
            "rate_limit": 1000,
            "created_at": datetime.now().isoformat()
        },
        {
            "key": "demo-user-key-67890",
            "name": "Demo User Key",
            "role": "user",
            "rate_limit": 100,
            "created_at": datetime.now().isoformat()
        }
    ]
    
    for key_data in demo_keys:
        API_KEYS[key_data["key"]] = key_data
        USAGE_STATS[key_data["key"]] = []
    
    logger.info(f"Initialized {len(demo_keys)} demo API keys")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Verify API key from Authorization header"""
    api_key = credentials.credentials
    
    if api_key not in API_KEYS:
        logger.warning(f"Invalid API key attempt: {api_key[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    key_data = API_KEYS[api_key]
    
    # Check rate limiting
    if not check_rate_limit(api_key, key_data["rate_limit"]):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    return key_data

def check_rate_limit(api_key: str, limit: int) -> bool:
    """Check if API key is within rate limit"""
    now = datetime.now()
    one_hour_ago = now - timedelta(hours=1)
    
    # Filter requests from last hour
    recent_requests = [
        req for req in USAGE_STATS.get(api_key, [])
        if datetime.fromisoformat(req["timestamp"]) > one_hour_ago
    ]
    
    USAGE_STATS[api_key] = recent_requests
    
    return len(recent_requests) < limit

def log_request(api_key: str, endpoint: str, model: str, response_time: float):
    """Log API request for monitoring"""
    if api_key not in USAGE_STATS:
        USAGE_STATS[api_key] = []
    
    USAGE_STATS[api_key].append({
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "model": model,
        "response_time": response_time
    })

def require_admin(key_data: Dict[str, Any] = Depends(verify_api_key)) -> Dict[str, Any]:
    """Require admin role"""
    if key_data["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return key_data

# API Endpoints

@app.get("/", tags=["General"])
async def root():
    """Root endpoint - API information"""
    return {
        "service": "Ollama API Service",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            ollama_status = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        ollama_status = "unreachable"
    
    return {
        "status": "healthy",
        "ollama_backend": ollama_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/keys", response_model=APIKeyResponse, tags=["API Keys"])
async def create_api_key(
    key_request: APIKeyCreate,
    admin: Dict[str, Any] = Depends(require_admin)
):
    """Create a new API key (admin only)"""
    import secrets
    
    # Generate secure API key
    api_key = f"ollama-{secrets.token_urlsafe(32)}"
    
    key_data = {
        "key": api_key,
        "name": key_request.name,
        "role": key_request.role,
        "rate_limit": key_request.rate_limit,
        "created_at": datetime.now().isoformat()
    }
    
    API_KEYS[api_key] = key_data
    USAGE_STATS[api_key] = []
    
    logger.info(f"Created new API key: {key_request.name} (role: {key_request.role})")
    
    return APIKeyResponse(
        api_key=api_key,
        name=key_data["name"],
        role=key_data["role"],
        rate_limit=key_data["rate_limit"],
        created_at=key_data["created_at"]
    )

@app.get("/api/keys", tags=["API Keys"])
async def list_api_keys(admin: Dict[str, Any] = Depends(require_admin)):
    """List all API keys (admin only)"""
    keys_list = []
    for api_key, data in API_KEYS.items():
        keys_list.append({
            "key_preview": f"{api_key[:20]}...",
            "name": data["name"],
            "role": data["role"],
            "rate_limit": data["rate_limit"],
            "created_at": data["created_at"]
        })
    
    return {"api_keys": keys_list, "total": len(keys_list)}

@app.delete("/api/keys/{key_preview}", tags=["API Keys"])
async def revoke_api_key(
    key_preview: str,
    admin: Dict[str, Any] = Depends(require_admin)
):
    """Revoke an API key (admin only)"""
    for api_key in list(API_KEYS.keys()):
        if api_key.startswith(key_preview):
            del API_KEYS[api_key]
            if api_key in USAGE_STATS:
                del USAGE_STATS[api_key]
            logger.info(f"Revoked API key: {key_preview}")
            return {"message": "API key revoked successfully"}
    
    raise HTTPException(status_code=404, detail="API key not found")

@app.get("/api/models", tags=["Ollama"])
async def list_models(key_data: Dict[str, Any] = Depends(verify_api_key)):
    """List available Ollama models"""
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            
            response_time = time.time() - start_time
            log_request(key_data["key"], "/api/models", "list", response_time)
            
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

@app.post("/api/generate", tags=["Ollama"])
async def generate(
    request: GenerateRequest,
    key_data: Dict[str, Any] = Depends(verify_api_key)
):
    """Generate text using Ollama model"""
    start_time = time.time()
    
    try:
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": request.stream,
        }
        
        if request.temperature is not None:
            payload["options"] = {"temperature": request.temperature}
        
        if request.max_tokens is not None:
            if "options" not in payload:
                payload["options"] = {}
            payload["options"]["num_predict"] = request.max_tokens
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            response_time = time.time() - start_time
            log_request(key_data["key"], "/api/generate", request.model, response_time)
            
            return response.json()
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

@app.post("/api/chat", tags=["Ollama"])
async def chat(
    request: ChatRequest,
    key_data: Dict[str, Any] = Depends(verify_api_key)
):
    """Chat with Ollama model"""
    start_time = time.time()
    
    try:
        messages = [msg.dict() for msg in request.messages]
        
        payload = {
            "model": request.model,
            "messages": messages,
            "stream": request.stream,
        }
        
        if request.temperature is not None:
            payload["options"] = {"temperature": request.temperature}
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload
            )
            response.raise_for_status()
            
            response_time = time.time() - start_time
            log_request(key_data["key"], "/api/chat", request.model, response_time)
            
            return response.json()
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

@app.get("/api/stats", response_model=UsageStats, tags=["Monitoring"])
async def get_usage_stats(key_data: Dict[str, Any] = Depends(verify_api_key)):
    """Get usage statistics for your API key"""
    stats = USAGE_STATS.get(key_data["key"], [])
    
    if not stats:
        return UsageStats(
            total_requests=0,
            requests_by_model={},
            requests_by_endpoint={},
            average_response_time=0.0,
            last_24h_requests=0
        )
    
    # Calculate statistics
    total_requests = len(stats)
    
    requests_by_model = {}
    requests_by_endpoint = {}
    total_response_time = 0.0
    
    now = datetime.now()
    last_24h = now - timedelta(hours=24)
    last_24h_count = 0
    
    for req in stats:
        # Count by model
        model = req.get("model", "unknown")
        requests_by_model[model] = requests_by_model.get(model, 0) + 1
        
        # Count by endpoint
        endpoint = req.get("endpoint", "unknown")
        requests_by_endpoint[endpoint] = requests_by_endpoint.get(endpoint, 0) + 1
        
        # Sum response times
        total_response_time += req.get("response_time", 0)
        
        # Count last 24h requests
        if datetime.fromisoformat(req["timestamp"]) > last_24h:
            last_24h_count += 1
    
    avg_response_time = total_response_time / total_requests if total_requests > 0 else 0.0
    
    return UsageStats(
        total_requests=total_requests,
        requests_by_model=requests_by_model,
        requests_by_endpoint=requests_by_endpoint,
        average_response_time=round(avg_response_time, 3),
        last_24h_requests=last_24h_count
    )

@app.get("/api/admin/stats", tags=["Monitoring"])
async def get_all_stats(admin: Dict[str, Any] = Depends(require_admin)):
    """Get usage statistics for all API keys (admin only)"""
    all_stats = {}
    
    for api_key, stats in USAGE_STATS.items():
        key_info = API_KEYS.get(api_key, {})
        all_stats[api_key[:20] + "..."] = {
            "key_name": key_info.get("name", "Unknown"),
            "role": key_info.get("role", "unknown"),
            "total_requests": len(stats),
            "last_request": stats[-1]["timestamp"] if stats else None
        }
    
    return {"statistics": all_stats}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
