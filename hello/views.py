import base64
import inspect
import math
import mimetypes
from pathlib import Path
import uuid
from urllib.parse import quote, urlencode

import cv2
import numpy as np
import requests
from django.conf import settings
from django.http import FileResponse, Http404, JsonResponse
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
SCALE_PARAM = "scale"
DEBUG_PARAM = "debug"
DEBUG_IMAGES_PARAM = "debug_images"
DEBUG_IMAGE_URLS_PARAM = "debug_image_urls"
DEBUG_IMAGE_FORMAT = ".png"
DEBUG_IMAGE_MIME_TYPE = "image/png"
SAVED_DEBUG_IMAGE_DEFAULT_MIME_TYPE = "application/octet-stream"
TRUE_QUERY_VALUES = {"1", "true", "yes", "on"}
EXAMPLE_IMAGE_URL = (
    "https://raw.githubusercontent.com/justingeeslin/Real-Time-Object-Measurement/"
    "main/test-images/ucard-one-off-axis/ucard-one-off-axis.jpg"
)


class BadReferenceSize(ValueError):
    pass


class BadMeasurementOption(ValueError):
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


def _parse_positive_int(value, param_name):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise BadMeasurementOption(f"'{param_name}' must be an integer.")

    if parsed <= 0:
        raise BadMeasurementOption(f"'{param_name}' must be greater than 0.")

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


def get_measurement_kwargs(query_params):
    kwargs = {"reference_size_mm": get_reference_size_mm(query_params)}

    scale = query_params.get(SCALE_PARAM)
    if scale is not None:
        kwargs["scale"] = _parse_positive_int(scale, SCALE_PARAM)

    return kwargs


def _query_param_enabled(query_params, param_name):
    value = query_params.get(param_name)
    if value is None:
        return False

    return str(value).strip().lower() in TRUE_QUERY_VALUES


def get_debug_image_root():
    return Path(settings.DEBUG_IMAGE_ROOT)


def build_debug_image_request_path():
    return get_debug_image_root() / uuid.uuid4().hex


def object_measurer_supports_saved_debug_images():
    try:
        parameters = inspect.signature(ObjectMeasurer).parameters
    except (TypeError, ValueError):
        return False

    if any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values()):
        return True

    return {"debug_path", "save_debug_images"}.issubset(parameters)


def _safe_debug_image_name(name):
    safe_name = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(name)
    ).strip("._")

    return safe_name or "debug_image"


def _debug_image_url_path(relative_path):
    prefix = settings.DEBUG_IMAGE_URL.strip("/")
    encoded_path = quote(relative_path.as_posix(), safe="/")
    return f"/{prefix}/{encoded_path}"


def _saved_debug_image_relative_path(image_path):
    root = get_debug_image_root().resolve()

    try:
        return Path(image_path).resolve().relative_to(root)
    except (OSError, ValueError):
        return None


def encode_saved_debug_image_urls(debug, request):
    urls = []

    for image in debug.get("debug_images", []):
        if not isinstance(image, dict) or not image.get("saved"):
            continue

        path = image.get("path")
        if not path:
            continue

        image_path = Path(path)
        relative_path = _saved_debug_image_relative_path(image_path)
        if relative_path is None:
            continue

        mime_type = (
            mimetypes.guess_type(str(image_path))[0]
            or SAVED_DEBUG_IMAGE_DEFAULT_MIME_TYPE
        )
        urls.append(
            {
                "name": str(image.get("name", "")),
                "filename": image_path.name,
                "mime_type": mime_type,
                "url": request.build_absolute_uri(
                    _debug_image_url_path(relative_path)
                ),
            }
        )

    return urls


def save_debug_image_arrays(debug, output_dir):
    output_dir = Path(output_dir)
    saved_images = []

    for name, value in debug.items():
        if name == "debug_images" or not _is_debug_image(value):
            continue

        filename = f"{len(saved_images)}_{_safe_debug_image_name(name)}{DEBUG_IMAGE_FORMAT}"
        out_path = output_dir / filename

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            saved = cv2.imwrite(str(out_path), value)
        except Exception:
            saved = False

        saved_images.append(
            {
                "name": str(name),
                "path": str(out_path),
                "saved": bool(saved),
            }
        )

    return saved_images


def ensure_debug_images_saved(debug, debug_path):
    if not debug_path:
        return

    saved_images = [
        image
        for image in debug.get("debug_images", [])
        if isinstance(image, dict) and image.get("saved")
    ]
    if saved_images:
        return

    fallback_images = save_debug_image_arrays(debug, debug_path)
    if fallback_images:
        debug["debug_images"] = fallback_images


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


def encode_debug(debug, exclude_keys=None):
    exclude_keys = set(exclude_keys or [])
    return {
        str(name): _serialize_debug_value(value)
        for name, value in debug.items()
        if str(name) not in exclude_keys
    }


def add_debug_response_fields(data, debug, query_params, request):
    if "object_contour_svg" in debug:
        data["svg"] = debug["object_contour_svg"]

    wants_debug_image_urls = _query_param_enabled(
        query_params,
        DEBUG_IMAGE_URLS_PARAM,
    )

    if (
        _query_param_enabled(query_params, DEBUG_PARAM)
        or _query_param_enabled(query_params, DEBUG_IMAGES_PARAM)
        or wants_debug_image_urls
    ):
        exclude_keys = {"debug_images"} if wants_debug_image_urls else set()
        data["debug"] = encode_debug(debug, exclude_keys=exclude_keys)

    if _query_param_enabled(query_params, DEBUG_IMAGES_PARAM):
        data["debug_images"] = encode_debug_images(debug)

    if wants_debug_image_urls:
        data["debug_image_urls"] = encode_saved_debug_image_urls(debug, request)


def serialize_measurement(measurement):
    serialized = {
        "height": measurement.height_cm,
        "width": measurement.width_cm,
    }

    bbox = getattr(measurement, "bbox", None)
    if bbox is not None:
        serialized["bbox"] = [int(value) for value in bbox]

    return serialized


def measure_with_debug(measurer, img):
    result = measurer.measure(img, return_debug=True)

    if isinstance(result, tuple) and len(result) == 2:
        measurements, debug = result
        return measurements, debug or {}

    return result, getattr(measurer, "debug", {})


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
        _build_example_link(
            request,
            "Return debug image URLs",
            "Saves debug images temporarily and returns public URLs for inspection.",
            {
                "url": EXAMPLE_IMAGE_URL,
                DEBUG_IMAGE_URLS_PARAM: "1",
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
    wants_debug_image_urls = _query_param_enabled(request.GET, DEBUG_IMAGE_URLS_PARAM)

    if not image_url:
        return index(request)

    try:
        measurement_kwargs = get_measurement_kwargs(request.GET)
    except BadReferenceSize as exc:
        return JsonResponse(
            {"error": "Invalid reference object dimensions", "details": str(exc)},
            status=400,
        )
    except BadMeasurementOption as exc:
        return JsonResponse(
            {"error": "Invalid measurement option", "details": str(exc)},
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

    # Construct the ObjectMeasurer with the same options used by direct tests.
    debug_image_request_path = None
    if wants_debug_image_urls:
        debug_image_request_path = build_debug_image_request_path()

    if (
        debug_image_request_path is not None
        and object_measurer_supports_saved_debug_images()
    ):
        measurement_kwargs["debug_path"] = str(debug_image_request_path)
        measurement_kwargs["save_debug_images"] = True

    measurer = ObjectMeasurer(**measurement_kwargs)

    data = {}

    try:
        # Get the measurements (in cm)
        measurements, debug = measure_with_debug(measurer, img)
    except Exception as exc:
        data["error"] = "Image measurement failed"
        data["details"] = str(exc)
        if wants_debug_image_urls:
            ensure_debug_images_saved(measurer.debug, debug_image_request_path)
        add_debug_response_fields(data, measurer.debug, request.GET, request)
        return JsonResponse(
            data,
            status=500,
        )

    if wants_debug_image_urls:
        ensure_debug_images_saved(debug, debug_image_request_path)

    add_debug_response_fields(data, debug, request.GET, request)

    if not measurements:
        data["error"] = "No measurable object found in image"
        return JsonResponse(
            data,
            status=422,
        )

    width_cm, height_cm = measurements[0].width_cm, measurements[0].height_cm

    data["height"] = height_cm
    data["width"] = width_cm
    data["measurements"] = [
        serialize_measurement(measurement) for measurement in measurements
    ]

    return JsonResponse(data)


def debug_image(request, path):
    relative_path = Path(path)
    image_path = get_debug_image_root() / relative_path

    if _saved_debug_image_relative_path(image_path) is None or not image_path.is_file():
        raise Http404("Debug image not found")

    content_type = (
        mimetypes.guess_type(str(image_path))[0]
        or SAVED_DEBUG_IMAGE_DEFAULT_MIME_TYPE
    )
    return FileResponse(image_path.open("rb"), content_type=content_type)


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
