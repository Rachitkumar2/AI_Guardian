import requests

url = "http://127.0.0.1:5000/detect"  # Ensure this matches the Flask endpoint
files = {"file": open("AI.wav", "rb")}
response = requests.post(url, files=files)

print(response.json())
