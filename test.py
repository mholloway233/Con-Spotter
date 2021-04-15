import pyrebase
from django.conf import settings
import pandas as pd


CONFIG = {
    "apiKey": "AIzaSyD9kyifI27SA7nT4zwHeCewFeiNSZ1dXdI",
    "authDomain": "con-spotter.firebaseapp.com",
    "databaseURL": "https://con-spotter-default-rtdb.firebaseio.com",
    "projectId": "con-spotter",
    "storageBucket": "con-spotter.appspot.com",
    "messagingSenderId": "251669384282",
    "appId": "1:251669384282:web:c807a8af700d1581795c63",
    "measurementId": "G-6YJY0FHLKC"
  }



firebase_storage = pyrebase.initialize_app(CONFIG)
storage = firebase_storage.storage()


URL = storage.child("creditcard.csv").get_url(None)
