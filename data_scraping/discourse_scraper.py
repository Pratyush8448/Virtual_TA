import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def get_discourse_topics():
    print("[INFO] Launching headless browser...")

    # Chrome options for headless mode
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # Start browser
    driver = webdriver.Chrome(options=options)

    try:
        url = "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34/l/latest"
        print(f"[INFO] Visiting {url}")
        driver.get(url)

        # Wait for the JavaScript to render
        time.sleep(5)

        print("[INFO] Extracting topic list...")
        data = driver.execute_script("return window.__initialState;")
        driver.quit()

        with open("latest_topics.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print("[SUCCESS] Scraped and saved latest_topics.json")

        topics = data["topicList"]["topics"]
        for t in topics:
            print(f"- [{t['id']}] {t['title']}")

    except Exception as e:
        driver.quit()
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    get_discourse_topics()
