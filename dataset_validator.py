import json
import statistics

# If you don't have tiktoken installed, run: pip install tiktoken
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text):
        return len(enc.encode(text))
except ImportError:
    print("Tiktoken not found. Falling back to word count for length estimation.")
    def count_tokens(text):
        return len(text.split())

file_path = "aegis_medical_dataset.jsonl"

errors = 0
token_counts = []
valid_examples = 0

print(f"--- Starting Validation for {file_path} ---")

with open(file_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print(f"❌ Line {line_num}: Invalid JSON format (not a valid JSON object).")
            errors += 1
            continue

        # 1. Check ChatML Structure
        if "messages" not in data:
            print(f"❌ Line {line_num}: Missing 'messages' key.")
            errors += 1
            continue

        messages = data["messages"]
        if len(messages) != 3:
            print(f"❌ Line {line_num}: Expected 3 messages, found {len(messages)}.")
            errors += 1
            continue

        # 2. Check Roles
        expected_roles = ["system", "user", "assistant"]
        actual_roles = [m.get("role") for m in messages]
        if actual_roles != expected_roles:
            print(f"❌ Line {line_num}: Incorrect roles. Expected {expected_roles}, got {actual_roles}.")
            errors += 1
            continue

        # 3. Verify Assistant Output is Valid JSON (Your Capstone Requirement)
        assistant_content = messages[2].get("content", "")
        try:
            parsed_report = json.loads(assistant_content)
        except json.JSONDecodeError:
            print(f"❌ Line {line_num}: Assistant output is not valid JSON string.")
            errors += 1
            continue

        # 4. Token Counting
        total_tokens = sum(count_tokens(m.get("content", "")) for m in messages)
        token_counts.append(total_tokens)
        valid_examples += 1

# Print Final Report
print("\n--- Validation Report ---")
print(f"Total Lines Processed: {line_num}")
print(f"Valid Examples: {valid_examples}")
print(f"Total Errors Found: {errors}")

if valid_examples > 0:
    print("\n--- Token Length Statistics ---")
    print(f"Min length: {min(token_counts)} tokens")
    print(f"Max length: {max(token_counts)} tokens")
    print(f"Mean length: {int(statistics.mean(token_counts))} tokens")
    print(f"Median length: {int(statistics.median(token_counts))} tokens")

    # 95th percentile
    sorted_tokens = sorted(token_counts)
    idx_95 = int(len(sorted_tokens) * 0.95)
    print(f"95th Percentile: {sorted_tokens[idx_95]} tokens")

if errors == 0 and valid_examples >= 200:
    print("\n✅ DATASET APPROVED. Ready for fine-tuning.")
else:
    print("\n⛔ DATASET FAILED. Fix the errors above before proceeding.")
