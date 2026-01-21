import requests
import base64
import matplotlib.pyplot as plt
import sys
import argparse
from io import BytesIO
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser(description="Test Anomaly Detection API Client")

    # Image Path Argument
    parser.add_argument(
        "--image_path",
        type=str,
        default="./data/carpet/test/color/000.png",
        help="Path to the image file to test",
    )

    # Server Connection Arguments
    parser.add_argument(
        "--host", type=str, default="localhost", help="API Host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=3000, help="API Port (default: 3000)"
    )

    return parser.parse_args()


def test_prediction(args):
    # Construct URL dynamically based on CLI args
    api_url = f"http://{args.host}:{args.port}/predict"

    # 1. Open the image file
    try:
        # We assume binary mode 'rb' is needed for file uploads
        file_obj = open(args.image_path, "rb")
    except FileNotFoundError:
        print(f"Error: Could not find image at {args.image_path}")
        sys.exit(1)

    print(f"Sending {args.image_path} to {api_url}...")

    # 2. Prepare Payload
    files = {"file": file_obj}

    # 3. Send POST request
    try:
        response = requests.post(api_url, files=files)
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to server at {api_url}. Is it running?")
        sys.exit(1)
    finally:
        # Good practice to close the file handle
        file_obj.close()

    # 4. Process Response
    if response.status_code == 200:
        data = response.json()
        print("\n--- Success! ---")
        print(f"Anomaly Score: {data['anomaly_score']}")
        print(f"Is Anomaly: {data['is_anomaly']}")

        # 5. Decode and Show Heatmap
        if data.get("heatmap_base64"):
            try:
                image_data = base64.b64decode(data["heatmap_base64"])
                img = Image.open(BytesIO(image_data))

                plt.figure(figsize=(5, 5))
                plt.imshow(img)
                plt.title(
                    f"Score: {data['anomaly_score']} | Anomaly: {data['is_anomaly']}"
                )
                plt.axis("off")
                plt.show()
            except Exception as e:
                print(f"Error displaying heatmap: {e}")
    else:
        print(f"Error {response.status_code}: {response.text}")


if __name__ == "__main__":
    args = get_args()
    test_prediction(args)
