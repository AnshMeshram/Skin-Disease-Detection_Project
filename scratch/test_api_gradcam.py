import requests
import base64
from pathlib import Path

def test_predict():
    url = "http://localhost:8080/predict"
    img_path = "raw/ISIC_2019_Training_Input/ISIC_0000000.jpg"
    if not Path(img_path).exists():
        print(f"Test image {img_path} not found")
        return

    with open(img_path, "rb") as f:
        files = {"file": f}
        try:
            r = requests.post(url, files=files)
            data = r.json()
            print(f"Status: {data.get('status')}")
            print(f"Prediction: {data.get('prediction')}")
            gradcam = data.get("gradcam")
            if gradcam:
                print(f"Grad-CAM received (len: {len(gradcam)})")
                with open("outputs/test_gradcam_api.png", "wb") as out:
                    out.write(base64.b64decode(gradcam))
                print("Saved Grad-CAM to outputs/test_gradcam_api.png")
            else:
                print("Grad-CAM is EMPTY or MISSING")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_predict()
