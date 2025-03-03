from collections import defaultdict

# 🔹 Group Data by Date → Channel → Type
grouped_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for entry in extracted_data:
    date = entry.get("Date", "Unknown")
    channel = entry.get("Channel", "Unknown")
    data_type = entry.get("Type", "Unknown")
    grouped_data[date][channel][data_type].append(entry)

# 🔹 Format Grouped Data for Iterative Processing (Max 4 Records per Chunk)
data_chunks = []

for date, channels in sorted(grouped_data.items()):  # Sorting by Date
    record_str = f"📅 **Date:** {date}\n"
    
    for channel, types in sorted(channels.items()):  # Sorting by Channel
        record_str += f"📌 **Channel:** {channel}\n"
        
        for data_type, records in sorted(types.items()):  # Sorting by Type
            record_str += f"📊 **Type:** {data_type}\n"

            for record in records:
                if record.get("Hourly Logins"):
                    record_str += f"⏰ Hourly Logins: {record['Hourly Logins']}\n"
                if record.get("Hourly Payments"):
                    record_str += f"💰 Hourly Payments: {record['Hourly Payments']}\n"
                record_str += "\n"  # Newline separator
            
            # 🔹 Append after 4 records to maintain chunk size
            if len(data_chunks) % 4 == 0:
                data_chunks.append(record_str)
                record_str = ""

# Append remaining records if any
if record_str:
    data_chunks.append(record_str)
