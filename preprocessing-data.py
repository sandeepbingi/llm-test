import json

def extract_unique_messages(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        logs = json.load(f)
    
    # Extract unique messages
    unique_messages = set()
    for log in logs:
        unique_messages.add(log['input']['message'])
    
    # Print unique messages
    print("Unique messages found in logs:")
    for message in unique_messages:
        print(f"- {message}")
    
    return unique_messages

def update_messages_with_mapping(file_path, output_file, message_output_map):
    with open(file_path, 'r', encoding='utf-8') as f:
        logs = json.load(f)
    
    # Update logs with provided output values
    for log in logs:
        message = log['input']['message']
        if message in message_output_map:
            log['output'] = message_output_map[message]
    
    # Save the updated logs
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=4)
    
    print(f"Updated logs saved to {output_file}")

# Example usage
if __name__ == "__main__":
    file_path = 'logs.json'
    output_file = 'updated_logs.json'
    unique_messages = extract_unique_messages(file_path)
    
    # User should manually map messages to output classifications and provide the dictionary
    message_output_map = {
        # Fill this dictionary with user-provided mappings
    }
    
    update_messages_with_mapping(file_path, output_file, message_output_map)
