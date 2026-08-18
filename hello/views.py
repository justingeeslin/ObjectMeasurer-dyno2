import base64
import math
from pathlib import Path
from urllib.parse import urlencode

import cv2
import numpy as np
import requests
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
DEBUG_PARAM = "debug"
DEBUG_IMAGES_PARAM = "debug_images"
DEBUG_IMAGE_FORMAT = ".png"
DEBUG_IMAGE_MIME_TYPE = "image/png"
TRUE_QUERY_VALUES = {"1", "true", "yes", "on"}
EXAMPLE_IMAGE_URL = (
    "https://raw.githubusercontent.com/justingeeslin/Real-Time-Object-Measurement/"
    "main/test-images/ucard-one-off-axis/ucard-one-off-axis.jpg"
)


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


def _serialize_debug_value(value):
    if isinstance(value, np.generic):
        return _serialize_debug_value(value.item())

    if isinstance(value, np.ndarray):
        height, width = value.shape[:2] if value.ndim >= 2 else (None, None)
        serialized = {
            "type": "image" if _is_debug_image(value) else "ndarray",
            "shape": [int(dimension) for dimension in value.shape],
            "dtype": str(value.dtype),
        }
        if height is not None and width is not None:
            serialized["width"] = int(width)
            serialized["height"] = int(height)
        return serialized

    if isinstance(value, dict):
        return {str(key): _serialize_debug_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_serialize_debug_value(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float) and not math.isfinite(value):
        return str(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def encode_debug(debug):
    return {str(name): _serialize_debug_value(value) for name, value in debug.items()}


def add_debug_response_fields(data, debug, query_params):
    if "object_contour_svg" in debug:
        data["svg"] = debug["object_contour_svg"]

    if (
        _query_param_enabled(query_params, DEBUG_PARAM)
        or _query_param_enabled(query_params, DEBUG_IMAGES_PARAM)
    ):
        data["debug"] = encode_debug(debug)

    if _query_param_enabled(query_params, DEBUG_IMAGES_PARAM):
        data["debug_images"] = encode_debug_images(debug)


def _build_example_link(request, title, description, params):
    query_string = urlencode(params)
    href = request.build_absolute_uri(f"/?{query_string}")

    return {
        "title": title,
        "description": description,
        "href": href,
        "path": f"/?{query_string}",
        "curl": f'curl "{href}"',
    }


def build_index_examples(request):
    return [
        _build_example_link(
            request,
            "Measure an object",
            "Uses the default portrait poster-board reference size.",
            {"url": EXAMPLE_IMAGE_URL},
        ),
        _build_example_link(
            request,
            "Use a letter-size reference",
            "Overrides the reference object dimensions in millimeters.",
            {
                "url": EXAMPLE_IMAGE_URL,
                REFERENCE_SIZE_PARAM: f"{LETTER_MM[0]},{LETTER_MM[1]}",
            },
        ),
        _build_example_link(
            request,
            "Include structured debug data",
            "Adds metadata, trace information, and image descriptors to the JSON.",
            {
                "url": EXAMPLE_IMAGE_URL,
                DEBUG_PARAM: "1",
            },
        ),
        _build_example_link(
            request,
            "Include debug images",
            "Adds base64-encoded PNG debug images that clients can render.",
            {
                "url": EXAMPLE_IMAGE_URL,
                DEBUG_IMAGES_PARAM: "1",
            },
        ),
    ]


def index(request):
    return render(
        request,
        "index.html",
        {
            "examples": build_index_examples(request),
            "example_image_url": EXAMPLE_IMAGE_URL,
            "default_reference_size": PORTRAIT_POSTER_BOARD_MM,
            "letter_reference_size": LETTER_MM,
        },
    )


def measure(request):
    image_url = request.GET.get("url")

    if not image_url:
        return index(request)

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

    data = {}

    try:
        # Get the measurements (in cm)
        measurements = measurer.measure(img)
    except Exception as exc:
        data["error"] = "Image measurement failed"
        data["details"] = str(exc)
        add_debug_response_fields(data, measurer.debug, request.GET)
        return JsonResponse(
            data,
            status=500,
        )

    add_debug_response_fields(data, measurer.debug, request.GET)

    if not measurements:
        data["error"] = "No measurable object found in image"
        return JsonResponse(
            data,
            status=422,
        )

    width_cm, height_cm = measurements[0].width_cm, measurements[0].height_cm

    data["height"] = height_cm
    data["width"] = width_cm

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
