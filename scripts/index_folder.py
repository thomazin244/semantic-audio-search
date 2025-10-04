import os
import sys
import requests

API = os.getenv("API_URL", "http://localhost:8000")
EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

def upload(path, tags="", description=""):
    filename = os.path.basename(path)
    with open(path, "rb") as f:
        files = {"file": (filename, f.read())}
    data = {"tags": tags, "description": description}
    r = requests.post(f"{API}/upload", files=files, data=data, timeout=600)
    r.raise_for_status()
    print(f"OK  {filename}")

def main(folder):
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        sys.exit(1)

    files = []
    for root, _, names in os.walk(folder):
        for n in names:
            if n.lower().endswith(EXTS):
                files.append(os.path.join(root, n))

    print(f"Found {len(files)} audio files")
    for i, p in enumerate(files, 1):
        try:
            upload(p)
        except Exception as e:
            print(f"ERR {p} -> {e}")
        if i % 50 == 0:
            print(f"... {i} files processed")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/index_folder.py /path/to/folder")
        sys.exit(1)
    main(sys.argv[1])
