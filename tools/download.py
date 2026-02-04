import requests
from pathlib import Path

input_file = "prompts.txt"
output_dir = Path("images")
output_dir.mkdir(exist_ok=True)

with open(input_file, "r") as f:
    urls = [line.strip() for line in f if line.strip()]

for url in urls:
    filename = url.split("/")[-1]  # keep original name
    filepath = output_dir / filename

    print(f"Downloading {url}")

    r = requests.get(url, timeout=30)
    r.raise_for_status()

    with open(filepath, "wb") as img:
        img.write(r.content)

print("Done.")
