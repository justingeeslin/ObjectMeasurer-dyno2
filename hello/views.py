import requests
from django.http import HttpResponse
from django.http import JsonResponse
from django.shortcuts import render

from .models import Greeting
import cv2
from ObjectMeasurer import ObjectMeasurer
import os

# SHORT SIDE / X-AXIS FIRST
# A4_MM = (210.0, 297.0)
LETTER_MM = (215.9, 279.4)  # 8.5in x 11in

def index(request):
    return HttpResponse(f"<h2>hi</h2")

def measure(request):
    image_url = request.GET.get("url")

    if not image_url:
        from urllib.parse import quote
        example_image_url = "https://raw.githubusercontent.com/justingeeslin/Real-Time-Object-Measurement/main/test-images/ucard-one-off-axis/ucard-one-off-axis.jpg"
        encoded = quote(image_url, safe="")
        return HttpResponse(
            f"<h2>Error: Missing 'url' query parameter.</h2>"
            f"<p>Example usage: <a href=\"/?url={encoded}\">/?url={example_image_url}</a></p>",
            status=400
        )

    import numpy as np

    r = requests.get(image_url)

    # Convert bytes to numpy array
    image_array = np.frombuffer(r.content, np.uint8)

    # Decode image with OpenCV
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Construct the ObjectMeasurer with the size of the reference object
    measurer = ObjectMeasurer(reference_size_mm=LETTER_MM)
    # Get the measurements (in cm)
    measurements = measurer.measure(img)

    data = {
        "measurements": measurements
    }

    if measurer.debug['object_contour_svg'] is not None:
        data["svg"] = measurer.debug['object_contour_svg']

    return JsonResponse(data)

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
