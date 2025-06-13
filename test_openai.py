import requests

url = "http://127.0.0.1:8000/api/"
payload = {
    "question": "What is the purpose of virtual teaching assistants?"
    # If needed:
    # "image": base64_encoded_image_string
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

print("[Status Code]:", response.status_code)
print("[Response]:", response.json())
