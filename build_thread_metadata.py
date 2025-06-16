import os
import json

input_dir = "downloaded_threads/"
output_file = "data_chunks/discourse_threads.json"

threads = []

for file_name in os.listdir(input_dir):
    if file_name.endswith(".json"):
        with open(os.path.join(input_dir, file_name), "r", encoding="utf-8") as f:
            data = json.load(f)
            try:
                post = data["post_stream"]["posts"][0]
                title = post["topic_slug"].replace("-", " ").title()
                url = f'https://discourse.onlinedegree.iitm.ac.in/t/{post["topic_slug"]}/{post["topic_id"]}'
                content = post["cooked"]

                threads.append({
                    "title": title,
                    "url": url,
                    "content": content
                })
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Save
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(threads, f, indent=2)

print(f"[✔] Saved {len(threads)} threads to {output_file}")
