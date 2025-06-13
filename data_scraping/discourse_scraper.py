import requests
import time
import json
from tqdm import tqdm

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34/l/latest.json"

CATEGORY_SLUG = "tools-in-data-science"
CATEGORY_ID = 15  # Confirm this by visiting the category page JSON e.g. /c/tools-in-data-science/15.json
OUTPUT_FILE = "tds_discourse_threads.json"

def get_topics():
    print("[INFO] Fetching topic list...")
    url = "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34/l/latest.json"
    headers = {
        "User-Agent": "Mozilla/5.0",  # disguise as browser
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data.get("topic_list", {}).get("topics", [])


def get_topic_posts(topic_id):
    url = f"{BASE_URL}/t/{topic_id}.json"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"[WARN] Failed to fetch topic {topic_id}")
        return []
    data = response.json()
    posts = data.get("post_stream", {}).get("posts", [])
    return [{"username": p["username"], "content": p["cooked"], "post_number": p["post_number"]} for p in posts]

def scrape_discourse():
    topics = get_topics()
    # continue scraping posts...


    print(f"[INFO] Scraping {len(topics)} topics...")

    for topic in tqdm(topics):
        topic_id = topic["id"]
        title = topic["title"]
        posts = get_topic_posts(topic_id)
        results.append({
            "topic_id": topic_id,
            "title": title,
            "posts": posts
        })
        time.sleep(1)  # Avoid rate-limiting

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved {len(results)} threads to {OUTPUT_FILE}")

if __name__ == "__main__":
    scrape_discourse()

