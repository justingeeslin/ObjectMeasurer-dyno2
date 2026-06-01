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
PORTRAIT_POSTER_BOARD_MM = (561.975, 711.2)

def index(request):
    return HttpResponse(f"<h2>hi</h2")

def measure(request):
    image_url = request.GET.get("url")

    if not image_url:
        from urllib.parse import quote
        example_image_url = "https://raw.githubusercontent.com/justingeeslin/Real-Time-Object-Measurement/main/test-images/ucard-one-off-axis/ucard-one-off-axis.jpg"
        encoded = quote(example_image_url, safe="")
        return HttpResponse(
            f"<h2>Error: Missing 'url' query parameter.</h2>"
            f"<p>Example usage: <a href=\"/?url={encoded}\">/?url={example_image_url}</a></p>",
            status=400
        )

    import numpy as np
    try:
        r = requests.get(
            image_url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "image/*,*/*;q=0.8",
            },
            timeout=15,
        )

        if not r.ok:
            return JsonResponse(
                {
                    "request_url": image_url,
                    "error": "Image server rejected download",
                    "remote_status": r.status_code,
                    "remote_content_type": r.headers.get("Content-Type"),
                    "remote_body": r.text[:500],
                },
                status=400,
            )

    except requests.exceptions.RequestException as exc:
        return JsonResponse(
            {"error": "Could not download image", "details": str(exc)},
            status=400,
        )

    # Convert bytes to numpy array
    image_array = np.frombuffer(r.content, np.uint8)

    # Decode image with OpenCV
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if img is None:
        return JsonResponse(
            {"error": "Downloaded file is not a valid image"},
            status=400,
        )

    # Construct the ObjectMeasurer with the size of the reference object
    measurer = ObjectMeasurer(reference_size_mm=PORTRAIT_POSTER_BOARD_MM)

    try:
        # Get the measurements (in cm)
        measurements = measurer.measure(img)
    except Exception as exc:
        return JsonResponse(
            {"error": "Image measurement failed", "details": str(exc)},
            status=500,
        )

    if not measurements:
        return JsonResponse(
            {"error": "No measurable object found in image"},
            status=422,
        )

    width_cm, height_cm = measurements[0].width_cm, measurements[0].height_cm

    data = {
        "height": height_cm,
        "width": width_cm,
    }

    if 'object_contour_svg' in measurer.debug:
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
