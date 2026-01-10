"""
Mojo Gateway - High-Performance LLM Inference API Gateway

A proof-of-concept implementation demonstrating:
- Mojo-native HTTP server using Lightbug
- MAX Engine integration for LLM inference
- SIMD-accelerated statistics and rate limiting
- Zero-overhead authentication and validation

Usage:
    magic run dev          # Development server
    magic run build        # Build optimized binary
    ./bin/gateway          # Run production server

Environment Variables:
    GATEWAY_HOST           # Host to bind (default: 0.0.0.0)
    GATEWAY_PORT           # Port to bind (default: 8080)
    JWT_SECRET             # Secret for JWT signing
    MODEL_PATH             # Path to model or HuggingFace model ID
    LOG_LEVEL              # Logging level (DEBUG, INFO, WARNING, ERROR)
"""

from lightbug_http import Server

from .router import GatewayRouter
from .utils.config import Config, get_config
from .inference.max_engine import initialize_inference_engine
from .handlers.health import set_server_start_time


fn print_banner():
    """Print the startup banner."""
    print("")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                                                              ║")
    print("║   🔥 MOJO GATEWAY - High-Performance LLM API Gateway 🔥      ║")
    print("║                                                              ║")
    print("║   Powered by Mojo + MAX Engine                               ║")
    print("║   https://github.com/your-org/ollama-api-gateway             ║")
    print("║                                                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("")


fn print_config(config: Config):
    """Print current configuration."""
    print("Configuration:")
    print("  Host:              " + config.host)
    print("  Port:              " + String(config.port))
    print("  Model Path:        " + config.model_path)
    print("  Rate Limit:        " + String(config.rate_limit_requests) + " req/hour")
    print("  Metrics Enabled:   " + ("Yes" if config.enable_metrics else "No"))
    print("  Log Level:         " + config.log_level)
    print("")


fn print_endpoints(host: String, port: Int):
    """Print available endpoints."""
    var base_url = "http://" + host + ":" + String(port)

    print("Available Endpoints:")
    print("")
    print("  Health & Status:")
    print("    GET  " + base_url + "/          Service info")
    print("    GET  " + base_url + "/health    Health check")
    print("    GET  " + base_url + "/ready     Readiness probe")
    print("    GET  " + base_url + "/live      Liveness probe")
    print("")
    print("  Inference (requires API key):")
    print("    POST " + base_url + "/api/generate         Text generation")
    print("    POST " + base_url + "/api/chat             Chat completions")
    print("    POST " + base_url + "/v1/chat/completions  OpenAI-compatible")
    print("")
    print("  Models:")
    print("    GET  " + base_url + "/api/models           List models")
    print("    GET  " + base_url + "/v1/models            OpenAI-compatible")
    print("")
    print("  Management (admin only):")
    print("    POST " + base_url + "/api/keys             Create API key")
    print("    GET  " + base_url + "/api/keys             List API keys")
    print("    DEL  " + base_url + "/api/keys/{id}        Revoke API key")
    print("")
    print("  Statistics:")
    print("    GET  " + base_url + "/api/stats            User statistics")
    print("    GET  " + base_url + "/api/stats/detailed   Detailed stats")
    print("    GET  " + base_url + "/api/admin/stats      Admin statistics")
    print("")


fn main() raises:
    """
    Main entry point for the Mojo Gateway server.
    """
    print_banner()

    # Load configuration
    var config = get_config()
    print_config(config)

    # Initialize inference engine
    print("Initializing inference engine...")
    try:
        initialize_inference_engine()
        print("✓ Inference engine ready")
    except e:
        print("⚠ Warning: Could not initialize inference engine: " + String(e))
        print("  Server will start but inference endpoints may not work.")
    print("")

    # Print available endpoints
    print_endpoints(config.host, config.port)

    # Set server start time for uptime tracking
    set_server_start_time()

    # Create router
    var router = GatewayRouter()

    # Create and start server
    var server = Server()
    var listen_address = config.get_listen_address()

    print("═" * 64)
    print("")
    print("🚀 Server starting on " + listen_address)
    print("")
    print("Press Ctrl+C to stop the server")
    print("")

    # Start listening (blocking)
    server.listen_and_serve(listen_address, router)


# Entry point
fn run():
    """Wrapper for main to handle errors gracefully."""
    try:
        main()
    except e:
        print("")
        print("❌ Fatal error: " + String(e))
        print("")
        print("Please check your configuration and try again.")
