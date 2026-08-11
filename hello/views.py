import base64
import math

import cv2
import numpy as np
import requests
from django.http import HttpResponse
from django.http import JsonResponse
from django.shortcuts import render
from ObjectMeasurer import ObjectMeasurer

from .models import Greeting

# SHORT SIDE / X-AXIS FIRST
# A4_MM = (210.0, 297.0)
LETTER_MM = (215.9, 279.4)  # 8.5in x 11in
PORTRAIT_POSTER_BOARD_MM = (561.975, 711.2)
REFERENCE_WIDTH_PARAM = "reference_width_mm"
REFERENCE_HEIGHT_PARAM = "reference_height_mm"
REFERENCE_SIZE_PARAM = "reference_size_mm"
DEBUG_IMAGES_PARAM = "debug_images"
DEBUG_IMAGE_FORMAT = ".png"
DEBUG_IMAGE_MIME_TYPE = "image/png"
TRUE_QUERY_VALUES = {"1", "true", "yes", "on"}


class BadReferenceSize(ValueError):
    pass


def _parse_positive_float(value, param_name):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise BadReferenceSize(f"'{param_name}' must be a number.")

    if not math.isfinite(parsed) or parsed <= 0:
        raise BadReferenceSize(
            f"'{param_name}' must be a finite number greater than 0."
        )

    return parsed


def get_reference_size_mm(query_params):
    compact_size = query_params.get(REFERENCE_SIZE_PARAM)
    width = query_params.get(REFERENCE_WIDTH_PARAM)
    height = query_params.get(REFERENCE_HEIGHT_PARAM)

    if compact_size is not None:
        if width is not None or height is not None:
            raise BadReferenceSize(
                "Use either 'reference_size_mm' or "
                "'reference_width_mm'/'reference_height_mm', not both."
            )

        parts = [part.strip() for part in compact_size.split(",")]
        if len(parts) != 2 or not all(parts):
            raise BadReferenceSize(
                "'reference_size_mm' must be two comma-separated numbers: width,height."
            )

        return (
            _parse_positive_float(parts[0], REFERENCE_SIZE_PARAM),
            _parse_positive_float(parts[1], REFERENCE_SIZE_PARAM),
        )

    if width is None and height is None:
        return PORTRAIT_POSTER_BOARD_MM

    if width is None or height is None:
        raise BadReferenceSize(
            "'reference_width_mm' and 'reference_height_mm' must be provided together."
        )

    return (
        _parse_positive_float(width, REFERENCE_WIDTH_PARAM),
        _parse_positive_float(height, REFERENCE_HEIGHT_PARAM),
    )


def _query_param_enabled(query_params, param_name):
    value = query_params.get(param_name)
    if value is None:
        return False

    return str(value).strip().lower() in TRUE_QUERY_VALUES


def _is_debug_image(value):
    if not isinstance(value, np.ndarray):
        return False

    if value.ndim == 2:
        return True

    return value.ndim == 3 and value.shape[2] in (1, 3, 4)


def encode_debug_images(debug):
    images = {}

    for name, value in debug.items():
        if not _is_debug_image(value):
            continue

        ok, encoded = cv2.imencode(DEBUG_IMAGE_FORMAT, value)
        if not ok:
            continue

        height, width = value.shape[:2]
        images[name] = {
            "mime_type": DEBUG_IMAGE_MIME_TYPE,
            "encoding": "base64",
            "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "width": int(width),
            "height": int(height),
            "shape": [int(dimension) for dimension in value.shape],
            "dtype": str(value.dtype),
        }

    return images


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

    try:
        reference_size_mm = get_reference_size_mm(request.GET)
    except BadReferenceSize as exc:
        return JsonResponse(
            {"error": "Invalid reference object dimensions", "details": str(exc)},
            status=400,
        )

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
    measurer = ObjectMeasurer(reference_size_mm=reference_size_mm)

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

    if _query_param_enabled(request.GET, DEBUG_IMAGES_PARAM):
        data["debug_images"] = encode_debug_images(measurer.debug)

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
