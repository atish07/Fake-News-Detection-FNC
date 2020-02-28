import requests
import os
import zipfile
from tqdm import tqdm


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def downloadfile():
    file_id = '14sLiOG07y7ea7j6W7iisa9M1k-Z-vKx2'
    destination = os.getcwd()+"/data.zip"
    print("Downloading...")
    download_file_from_google_drive(file_id, destination)
    print("Extracting...")
    with zipfile.ZipFile("data.zip", "r") as zip_ref:
        zip_ref.extractall("targetdir")
