import os
import re
import requests
from urllib.parse import urljoin

BASE_URL = "https://tds.s-anand.net/2025-01/"
SIDEBAR_URL = urljoin(BASE_URL, "_sidebar.md")
OUTPUT_FOLDER = "data_scraping/course_modules"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

response = requests.get(SIDEBAR_URL)
if response.status_code != 200:
    print(f"Failed to fetch sidebar: {response.status_code}")
    exit()

sidebar_content = response.text

pattern = r'- \[([^\]]+)\]\(([^)]+\.md)\)'
matches = re.findall(pattern, sidebar_content)

modules = [(title, link) for (title, link) in matches if not link.startswith('_') and link != "README.md"]

def clean_filename(name):
    # Replace invalid chars for filenames with underscore
    return re.sub(r'[\\/:"*?<>|]+', '_', name).lower().replace(' ', '_')

for title, link in modules:
    full_url = urljoin(BASE_URL, link)

    clean_title = clean_filename(title)
    file_path = os.path.join(OUTPUT_FOLDER, f"{clean_title}.md")

    res = requests.get(full_url)
    if res.status_code == 200:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(res.text)
        print(f"✅ Saved '{title}' as {clean_title}.md")
    else:
        print(f"❌ Failed: {title} | Status: {res.status_code}")
