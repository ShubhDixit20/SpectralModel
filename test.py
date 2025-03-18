# https://spectralmodel.onrender.com

import requests

url = "https://spectralmodel.onrender.com/predict"

data = {"wavelengths": [2.4, 2.45, 2.67, 2.81, 4.58, 10.02, 12.46, 11.07, 8.45, 7.26, 
                         5.48, 3.85, 3.84, 3.76, 4.32, 7.18, 8.23, 7.15]}

response = requests.post(url, json=data)
print(response.json())