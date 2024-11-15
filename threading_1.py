import threading
import requests

def download_file(url):
    response = requests.get(url)
    print(f"Downloaded {url} with status code {response.status_code}")

urls = ["http://example.com/file1", "http://example.com/file2", "http://example.com/file3"]

threads = []
for url in urls:
    thread = threading.Thread(target=download_file, args=(url,))
    thread.start()
    threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()

print("All files downloaded.")


import threading

def read_file(file_path):
    with open(file_path, "r") as file:
        data = file.read()
        print(f"Read from {file_path}: {data[:100]}...")

files = ["file1.txt", "file2.txt", "file3.txt"]

threads = []
for file in files:
    thread = threading.Thread(target=read_file, args=(file,))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()

print("All files read.")
