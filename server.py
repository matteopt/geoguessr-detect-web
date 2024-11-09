from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import re
from PIL import Image
import io
import csv
from transformers import CLIPProcessor, CLIPModel
import json
import requests

model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

cities = {}

def init_data():
    data = requests.get("https://github.com/dr5hn/countries-states-cities-database/raw/refs/heads/master/csv/cities.csv").text
    csvr = csv.reader(data, delimiter=",", quotechar='"')
    _cities = list(csvr)

    global cities
    for l in _cities:
        country = l[7]
        state = l[4]
        city = l[1]
        if country not in cities:
            cities[country] = {}
        if state not in cities[country]:
            cities[country][state] = set()
        cities[country][state].add(city)

    len_countries = len(cities)
    len_states = sum(len(state) for state in cities.values())
    len_cities = len(_cities)
    logging.info(f"Loaded {len_countries} countries, {len_states} states, {len_cities} cities")

def guess(image, choices):
    inputs = processor(text=choices, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

    lprobs = []
    for i in range(len(choices)):
        lprobs.append((choices[i], probs.tolist()[0][i]))

    lprobs = sorted(lprobs, key=lambda p: p[1])
    print("\n".join([f"{p[1]} {p[0]}" for p in lprobs[-5:]]))
    return lprobs

class HTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if re.search("/country", self.path):
            length = int(self.headers.get("content-length"))
            data = self.rfile.read(length)

            image = Image.open(io.BytesIO(data))
            choices = list(cities.keys()) # countries
            lprobs = guess(image, choices)
            j = json.dumps(lprobs)

            self.send_response(200)
            self.end_headers()
            self.wfile.write(j.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def main():
    logging.basicConfig(level=logging.INFO)

    init_data()
    
    host = "localhost"
    port = 8000
    server = HTTPServer((host, port), HTTPRequestHandler)
    logging.info(f"HTTP server listening on {host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
    logging.info("Stopped HTTP server")

if __name__ == "__main__":
    main()
