import requests
from django.http import HttpResponse
from django.shortcuts import render

from .models import Greeting
import cv2
from ObjectMeasurer import ObjectMeasurer
import os

# SHORT SIDE / X-AXIS FIRST
# A4_MM = (210.0, 297.0)
LETTER_MM = (215.9, 279.4)  # 8.5in x 11in

def index(request):
    image_url = "https://raw.githubusercontent.com/justingeeslin/Real-Time-Object-Measurement/main/test-images/ucard-one-off-axis/ucard-one-off-axis.jpg"
    image_name = "ucard-one-off-axis.jpg"

    # import requests
    import numpy as np
    # import cv2

    # url = "https://example.com/image.jpg"

    r = requests.get(image_url)

    # Convert bytes to numpy array
    image_array = np.frombuffer(r.content, np.uint8)

    # Decode image with OpenCV
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Construct the ObjectMeasurer with the size of the reference object
    measurer = ObjectMeasurer(reference_size_mm=LETTER_MM)
    # Get the measurements (in cm)
    measurements = measurer.measure(img)

    return HttpResponse(f'<h1>{[(m.width_cm, m.height_cm) for m in measurements]}</h1>')

def db(request):
    # If you encounter errors visiting the `/db/` page on the example app, check that:
    #
    # When running the app on Heroku:
    #   1. You have added the Postgres database to your app.
    #   2. You have uncommented the `psycopg` dependency in `requirements.txt`, and the `release`
    #      process entry in `Procfile`, git committed your changes and re-deployed the app.
    #
    # When running the app locally:
    #   1. You have run `./manage.py migrate` to create the `hello_greeting` database table.

    greeting = Greeting()
    greeting.save()

    greetings = Greeting.objects.all()

    return render(request, "db.html", {"greetings": greetings})
