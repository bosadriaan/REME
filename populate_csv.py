import requests
import csv

# URL of your FastAPI server
url = "http://127.0.0.1:8000/documents"

# Function to add a document
def add_document(sentence_data):
    response = requests.post(url, json={
        "sentence": sentence_data["sentence"],
        "metadata": {
            "language": sentence_data["language"],
            "country": sentence_data["country"],
            "user_id": sentence_data["user_id"],
            "time": sentence_data["time"]
        }
    })
    if response.status_code == 200:
        print(f"Document added successfully: {response.json()}")
    else:
        print(f"Failed to add document: {response.text}")

# Function to read data from CSV and add documents
def populate_data_from_csv(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            add_document(row)

# Specify the path to your CSV file
csv_file_path = 'data.csv'

# Call the function to read data from CSV and add documents
populate_data_from_csv(csv_file_path)
