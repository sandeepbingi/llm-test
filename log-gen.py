import json
import random
import uuid
from datetime import datetime, timedelta

# Log Scenarios
SCENARIOS = {
    "Database Issues": [
        ("INFO", "Database query executed successfully."),
        ("INFO", "Database connection established."),
        ("INFO", "Slow query detected, execution time exceeding threshold."),
        ("INFO", "Repeated slow queries impacting performance."),
        ("ERROR", "Database timeout encountered."),
        ("ERROR", "Failed to establish a connection to the database."),
        ("ERROR", "Database crashed! Service unavailable.")
    ],
    "API Failures": [
        ("INFO", "API responded successfully."),
        ("INFO", "API latency within normal limits."),
        ("INFO", "Slight increase in API response time."),
        ("INFO", "API latency higher than expected."),
        ("ERROR", "API returned multiple 5xx errors."),
        ("ERROR", "API requests failing intermittently."),
        ("ERROR", "API completely unresponsive!")
    ],
    "Authentication Problems": [
        ("INFO", "User login successful."),
        ("INFO", "Token validation successful."),
        ("INFO", "Login taking slightly longer than usual."),
        ("INFO", "Multiple token expirations detected."),
        ("ERROR", "Repeated authentication failures."),
        ("ERROR", "Account locked due to multiple failed attempts."),
        ("ERROR", "System-wide authentication outage!")
    ],
    "Payment Processing Failures": [
        ("INFO", "Payment processed successfully."),
        ("INFO", "Transaction completed without issues."),
        ("INFO", "Delayed response from payment gateway."),
        ("INFO", "Higher than usual transaction failure rate."),
        ("ERROR", "Payment gateway errors detected."),
        ("ERROR", "Multiple transaction failures reported."),
        ("ERROR", "Payment processing system down!")
    ],
    "Cache Failures": [
        ("INFO", "Cache hit, data retrieved successfully."),
        ("INFO", "Cache performance is optimal."),
        ("INFO", "Increased cache misses observed."),
        ("INFO", "High memory usage detected in cache."),
        ("ERROR", "Cache eviction due to memory constraints."),
        ("ERROR", "Cache unresponsive, affecting performance."),
        ("ERROR", "Cache failure causing service degradation!")
    ],
    "Infrastructure Issues": [
        ("INFO", "Server running within normal parameters."),
        ("INFO", "CPU and memory usage within expected limits."),
        ("INFO", "CPU usage spiking above 80%."),
        ("INFO", "Memory leak detected in application."),
        ("ERROR", "Node failure detected, service degraded."),
        ("ERROR", "Multiple nodes reporting high failure rates."),
        ("ERROR", "System outage due to critical infrastructure failure!")
    ],
    "Internal Server Errors": [
        ("INFO", "Server responded with 200 OK."),
        ("INFO", "Server processing requests normally."),
        ("INFO", "Occasional slow server response."),
        ("INFO", "Increased load on the server."),
        ("ERROR", "HTTP 500 Internal Server Error encountered."),
        ("ERROR", "Multiple HTTP 500 errors observed."),
        ("ERROR", "Complete server failure, returning 500 for all requests!")
    ],
    "Response Time Issues": [
        ("INFO", "Response time within normal limits."),
        ("INFO", "Slight increase in response time."),
        ("INFO", "Response time exceeding threshold."),
        ("INFO", "Multiple slow responses detected."),
        ("ERROR", "Severe response time delays."),
        ("ERROR", "Service unavailable due to high response time."),
        ("ERROR", "System unable to process requests within SLA limits!")
    ],
    "Client-Side Errors": [
        ("INFO", "Client request successful."),
        ("INFO", "Client request validation passed."),
        ("INFO", "Few client requests with invalid parameters."),
        ("INFO", "Repeated 4xx client errors observed."),
        ("ERROR", "HTTP 400 Bad Request."),
        ("ERROR", "Multiple HTTP 404 Not Found errors."),
        ("ERROR", "High volume of client-side errors reported!")
    ]
}

# Constants
CHANNELS = ["CPL", "IPL", "CCR", "B2B", "B2C"]
SPRING_PROFILES = ["prod-dal", "prod-phx"]
SPACE_NAMES = ["api-APPS-green", "api-APPS-blue", "api-SERVICES-red", "api-SERVICES-yellow"]

# Number of logs
NUM_LOGS = 10000

# Generate logs
logs = []
start_time = datetime.now()

for _ in range(NUM_LOGS):
    scenario = random.choice(list(SCENARIOS.keys()))
    log_level, message = random.choice(SCENARIOS[scenario])

    # Generate realistic exception messages for ERROR logs
    if log_level == "ERROR":
        exception_msg = (
            f"{scenario.replace(' ', '')}Exception: {message}\n"
            f"at com.example.{scenario.replace(' ', '')}Service.processRequest({scenario.replace(' ', '')}Service.java:42)\n"
            f"at com.example.{scenario.replace(' ', '')}Controller.handleRequest({scenario.replace(' ', '')}Controller.java:27)\n"
            f"Caused by: java.lang.NullPointerException: {message}\n"
        )
    else:
        exception_msg = "NA"

    log_entry = {
        "event_time": (start_time - timedelta(seconds=random.randint(0, 1000000))).strftime("%Y-%m-%d %H:%M:%S"),
        "log_level": log_level,
        "channel_type": random.choice(CHANNELS),
        "app_name": f"{scenario.replace(' ', '_').lower()}_service",
        "space_name": random.choice(SPACE_NAMES),
        "spring_profile": random.choice(SPRING_PROFILES),
        "calling_API": f"/api/{scenario.replace(' ', '_').lower()}",
        "class_name": f"{scenario.replace(' ', '')}Handler",
        "method_name": "process_request",
        "exception": exception_msg,
        "message": message,
        "correlation_id": str(uuid.uuid4()),
        "transaction_id": str(uuid.uuid4())
    }

    logs.append(log_entry)

# Save to JSON file
file_path = "generated_logs_10k.json"
with open(file_path, "w") as f:
    json.dump(logs, f, indent=4)

print(f"✅ Successfully generated {NUM_LOGS} logs and saved to {file_path}")
