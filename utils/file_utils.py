import requests
from datetime import datetime


def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def generate_time_based_string():
    now = datetime.now()
    string = now.strftime("%Y%m%d%H%M%S%f")
    return string