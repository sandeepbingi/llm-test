import random
import time
import uuid
import json
from datetime import datetime

# Define the six scenarios with a progression from INFO to ERROR.
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
    ]
}

# Predefined lists for the various keys
APPS = ["PaymentService", "AuthService", "OrderService", "InventoryService"]
CHANNEL_TYPES = ["CPL", "IPL", "CCR"]
SPACE_NAMES = ["api-Payment-green", "api-Auth-green", "api-Order-green", "api-Inventory-green"]
SPRING_PROFILES = ["prod-dal", "prod-phx"]
CALLING_APIS = ["/login", "/order", "/payment", "/inventory"]
CLASS_NAMES = ["LoginController", "OrderController", "PaymentGateway", "InventoryManager"]
METHOD_NAMES = ["authenticateUser", "processOrder", "processPayment", "updateStock"]

# Exception mapping for error logs with realistic simulated Java stack traces.
EXCEPTIONS = {
    "Database Issues": {
         "Database timeout encountered.": (
             "java.sql.SQLTimeoutException: Query timed out after 30000ms\n"
             "\tat com.example.database.QueryExecutor.execute(QueryExecutor.java:85)\n"
             "\tat com.example.database.DatabaseConnection.run(DatabaseConnection.java:42)"
         ),
         "Failed to establish a connection to the database.": (
             "java.sql.SQLException: Unable to establish connection\n"
             "\tat com.example.database.ConnectionManager.getConnection(ConnectionManager.java:60)\n"
             "\tat com.example.database.DatabaseConnector.connect(DatabaseConnector.java:47)"
         ),
         "Database crashed! Service unavailable.": (
             "java.lang.RuntimeException: Database crash encountered\n"
             "\tat com.example.database.Database.run(Database.java:102)\n"
             "\tat com.example.Main.main(Main.java:14)"
         )
    },
    "API Failures": {
         "API returned multiple 5xx errors.": (
             "java.lang.RuntimeException: 5xx Server Error\n"
             "\tat com.example.api.ApiHandler.handle(ApiHandler.java:78)\n"
             "\tat com.example.api.ApiServer.process(ApiServer.java:55)"
         ),
         "API requests failing intermittently.": (
             "java.net.SocketTimeoutException: Read timed out\n"
             "\tat com.example.api.ApiHandler.handle(ApiHandler.java:102)\n"
             "\tat com.example.api.ApiServer.process(ApiServer.java:77)"
         ),
         "API completely unresponsive!": (
             "java.lang.IllegalStateException: API not responding\n"
             "\tat com.example.api.ApiHandler.handle(ApiHandler.java:150)\n"
             "\tat com.example.api.ApiServer.process(ApiServer.java:98)"
         )
    },
    "Authentication Problems": {
         "Repeated authentication failures.": (
             "java.lang.SecurityException: Multiple authentication failures\n"
             "\tat com.example.auth.AuthManager.authenticate(AuthManager.java:67)\n"
             "\tat com.example.auth.AuthServer.login(AuthServer.java:45)"
         ),
         "Account locked due to multiple failed attempts.": (
             "java.lang.IllegalAccessException: Account locked\n"
             "\tat com.example.auth.AccountManager.lockAccount(AccountManager.java:32)\n"
             "\tat com.example.auth.AuthServer.login(AuthServer.java:50)"
         ),
         "System-wide authentication outage!": (
             "java.lang.RuntimeException: Authentication system down\n"
             "\tat com.example.auth.AuthManager.authenticate(AuthManager.java:90)\n"
             "\tat com.example.auth.AuthServer.login(AuthServer.java:65)"
         )
    },
    "Payment Processing Failures": {
         "Payment gateway errors detected.": (
             "java.lang.RuntimeException: Payment gateway error\n"
             "\tat com.example.payment.PaymentGateway.process(PaymentGateway.java:40)\n"
             "\tat com.example.payment.PaymentService.handle(PaymentService.java:25)"
         ),
         "Multiple transaction failures reported.": (
             "java.lang.RuntimeException: Transaction failure\n"
             "\tat com.example.payment.PaymentGateway.process(PaymentGateway.java:55)\n"
             "\tat com.example.payment.PaymentService.handle(PaymentService.java:30)"
         ),
         "Payment processing system down!": (
             "java.lang.RuntimeException: Payment system not responding\n"
             "\tat com.example.payment.PaymentGateway.process(PaymentGateway.java:80)\n"
             "\tat com.example.payment.PaymentService.handle(PaymentService.java:45)"
         )
    },
    "Cache Failures": {
         "Cache eviction due to memory constraints.": (
             "java.lang.OutOfMemoryError: Java heap space\n"
             "\tat com.example.cache.CacheManager.evict(CacheManager.java:33)\n"
             "\tat com.example.cache.CacheService.clean(CacheService.java:21)"
         ),
         "Cache unresponsive, affecting performance.": (
             "java.lang.RuntimeException: Cache system failure\n"
             "\tat com.example.cache.CacheManager.get(CacheManager.java:58)\n"
             "\tat com.example.cache.CacheService.fetch(CacheService.java:40)"
         ),
         "Cache failure causing service degradation!": (
             "java.lang.RuntimeException: Cache crashed\n"
             "\tat com.example.cache.CacheManager.get(CacheManager.java:72)\n"
             "\tat com.example.cache.CacheService.fetch(CacheService.java:48)"
         )
    },
    "Infrastructure Issues": {
         "Node failure detected, service degraded.": (
             "java.lang.RuntimeException: Node failure\n"
             "\tat com.example.infra.Node.check(Node.java:89)\n"
             "\tat com.example.infra.Cluster.monitor(Cluster.java:56)"
         ),
         "Multiple nodes reporting high failure rates.": (
             "java.lang.RuntimeException: Cluster instability detected\n"
             "\tat com.example.infra.Cluster.checkNodes(Cluster.java:102)\n"
             "\tat com.example.infra.Cluster.monitor(Cluster.java:67)"
         ),
         "System outage due to critical infrastructure failure!": (
             "java.lang.RuntimeException: Critical infrastructure failure\n"
             "\tat com.example.infra.System.check(System.java:120)\n"
             "\tat com.example.Main.main(Main.java:15)"
         )
    }
}

def generate_log(scenario_name, log_level, message):
    """Generate a structured log entry in JSON format with all required keys."""
    # Determine the exception field:
    if log_level == "ERROR":
        # Look up a simulated exception stack trace; if not found, default to "NA"
        exception_detail = EXCEPTIONS.get(scenario_name, {}).get(message, "NA")
    else:
        exception_detail = "NA"
    
    log_entry = {
        "event_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "log_level": log_level,
        "channel_type": random.choice(CHANNEL_TYPES) or "NA",
        "app_name": random.choice(APPS) or "NA",
        "space_name": random.choice(SPACE_NAMES) or "NA",
        "spring_profile": random.choice(SPRING_PROFILES) or "NA",
        "calling_API": random.choice(CALLING_APIS) or "NA",
        "class_name": random.choice(CLASS_NAMES) or "NA",
        "method_name": random.choice(METHOD_NAMES) or "NA",
        "exception": exception_detail,
        "message": message,
        "correlation_id": str(uuid.uuid4()),
        "transaction_id": str(uuid.uuid4())
    }
    
    return json.dumps(log_entry, indent=4)

# Generate logs for each failure scenario
if __name__ == "__main__":
    for scenario, logs in SCENARIOS.items():
        print(f"\n### {scenario} ###\n")
        for log_level, message in logs:
            print(generate_log(scenario, log_level, message))
            time.sleep(1)  # Simulate log progression over time



SCENARIOS.update({
    "Internal Server Errors": [
        ("INFO", "Service is running normally."),
        ("INFO", "All systems operational."),
        ("INFO", "Unexpected internal error detected."),
        ("ERROR", "500 Internal Server Error: Unhandled exception."),
        ("ERROR", "500 Internal Server Error: Database connection failure."),
        ("ERROR", "500 Internal Server Error: Service unavailable due to overload."),
    ],
    "Response Time Issues": [
        ("INFO", "Server response time within normal limits."),
        ("INFO", "Response time slightly above normal threshold."),
        ("INFO", "Response time approaching critical threshold."),
        ("ERROR", "Response time exceeded acceptable limits."),
        ("ERROR", "API response time > 5 seconds."),
        ("ERROR", "Server response delay causing service degradation."),
    ],
    "Client Errors (400s)": [
        ("INFO", "Valid request received."),
        ("INFO", "Request parameters within acceptable range."),
        ("ERROR", "400 Bad Request: Invalid parameters."),
        ("ERROR", "400 Bad Request: Missing required fields."),
        ("ERROR", "400 Bad Request: Invalid request format."),
    ],
    "Service Unavailable": [
        ("INFO", "Service is available and functional."),
        ("INFO", "Service under maintenance, scheduled downtime."),
        ("INFO", "Service still available during maintenance window."),
        ("ERROR", "503 Service Unavailable: Server temporarily unavailable."),
        ("ERROR", "503 Service Unavailable: Overload or maintenance."),
        ("ERROR", "503 Service Unavailable: Server under heavy load, please try again."),
    ],
    "Timeouts": [
        ("INFO", "All requests completed successfully."),
        ("INFO", "Request completed within acceptable timeout."),
        ("INFO", "Request time approaching timeout."),
        ("ERROR", "Request timed out after 30 seconds."),
        ("ERROR", "Connection timeout with remote server."),
        ("ERROR", "Network timeout, service unavailable."),
    ]
})

# Now we will expand the `generate_log` method to include more specific scenarios.
def generate_log(scenario_name, log_level, message):
    """Generate a structured log entry in JSON format with all required keys."""
    # Determine the exception field:
    if log_level == "ERROR":
        # Look up a simulated exception stack trace; if not found, default to "NA"
        exception_detail = EXCEPTIONS.get(scenario_name, {}).get(message, "NA")
    else:
        exception_detail = "NA"
    
    log_entry = {
        "event_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "log_level": log_level,
        "channel_type": random.choice(CHANNEL_TYPES) or "NA",
        "app_name": random.choice(APPS) or "NA",
        "space_name": random.choice(SPACE_NAMES) or "NA",
        "spring_profile": random.choice(SPRING_PROFILES) or "NA",
        "calling_API": random.choice(CALLING_APIS) or "NA",
        "class_name": random.choice(CLASS_NAMES) or "NA",
        "method_name": random.choice(METHOD_NAMES) or "NA",
        "exception": exception_detail,
        "message": message,
        "correlation_id": str(uuid.uuid4()),
        "transaction_id": str(uuid.uuid4())
    }
    
    return json.dumps(log_entry, indent=4)

# Example for printing logs from the new failure scenarios
if __name__ == "__main__":
    for scenario, logs in SCENARIOS.items():
        print(f"\n### {scenario} ###\n")
        for log_level, message in logs:
            print(generate_log(scenario, log_level, message))
            time.sleep(1)  # Simulate log progression over time
