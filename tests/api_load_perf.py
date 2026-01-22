from locust import HttpUser, task, between
import io
import numpy as np
from PIL import Image


class AnomalyDetectionUser(HttpUser):
    # Simulate users waiting between 1 and 5 seconds between requests
    wait_time = between(1, 5)

    def on_start(self):
        """
        Generate a valid dummy image once when the user starts.
        This avoids overhead during the actual stress test.
        """
        data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(data)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.image_data = buf.getvalue()

    @task
    def predict(self):
        """
        Simulates a user hitting the endpoint.
        """
        # We must reset the buffer position or re-wrap bytes for every request
        files = {"file": ("stress_test.png", io.BytesIO(self.image_data), "image/png")}

        # Send POST request
        self.client.post("/predict", files=files)
