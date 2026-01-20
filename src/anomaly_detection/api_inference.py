# test_api.py
import requests
import base64
import matplotlib.pyplot as plt
import sys
from io import BytesIO
from PIL import Image

# 1. Configuration
API_URL = "http://localhost:3000/predict"
# Change this to an actual image path on your computer
IMAGE_PATH = "./data/carpet/test/color/000.png"


def test_prediction():
    # 2. Open the image file
    try:
        files = {"file": open(IMAGE_PATH, "rb")}
    except FileNotFoundError:
        print(f"Error: Could not find image at {IMAGE_PATH}")
        sys.exit(1)

    print(f"Sending {IMAGE_PATH} to {API_URL}...")

    # 3. Send POST request
    try:
        response = requests.post(API_URL, files=files)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Is it running?")
        sys.exit(1)

    # 4. Process Response
    if response.status_code == 200:
        data = response.json()
        print("\n--- Success! ---")
        print(f"Filename: {data['filename']}")
        print(f"Anomaly Score: {data['anomaly_score']}")
        print(f"Is Anomaly: {data['is_anomaly']}")

        # 5. Decode and Show Heatmap
        if data["heatmap_base64"]:
            image_data = base64.b64decode(data["heatmap_base64"])
            img = Image.open(BytesIO(image_data))

            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.title(f"Score: {data['anomaly_score']} | Anomaly: {data['is_anomaly']}")
            plt.axis("off")
            plt.show()
    else:
        print(f"Error {response.status_code}: {response.text}")


if __name__ == "__main__":
    test_prediction()
