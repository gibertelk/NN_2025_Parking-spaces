import requests
from pathlib import Path
from PIL import Image
import sys

def detect_parking(image_path: str, api_url: str = "http://localhost:8000"):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{api_url}/detect", files=files)

    return response.content


if __name__ == "__main__":

    image_path = "C:/Users/niker/OneDrive/Рабочий стол/11/test_000270.jpg"
    print(f"\nОбработка: {image_path}")
    
    try:
        result = detect_parking(image_path)
        
        # Сохранение результата
        output_path = f"{Path(image_path).stem}_detected.png"
        with open(output_path, 'wb') as f:
            f.write(result)

    except Exception as e:
        print(f"✗ Ошибка: {e}")
        sys.exit(1)