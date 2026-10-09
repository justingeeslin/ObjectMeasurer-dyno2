import base64
from dataclasses import dataclass
import html
import inspect
import logging
import math
import mimetypes
from pathlib import Path
import re
import tempfile
import uuid
from urllib.parse import quote, urlencode

import cv2
import numpy as np
import requests
from django.conf import settings
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from ObjectMeasurer import ObjectMeasurer

try:
    from ReferenceSurfaceMeasurer import ReferenceSurfaceMeasurer
except ImportError:  # pragma: no cover - optional until requirements are installed
    ReferenceSurfaceMeasurer = None

from .models import Greeting

logger = logging.getLogger(__name__)

# SHORT SIDE / X-AXIS FIRST
# A4_MM = (210.0, 297.0)
LETTER_MM = (215.9, 279.4)  # 8.5in x 11in
PORTRAIT_POSTER_BOARD_MM = (561.975, 711.2)
REFERENCE_WIDTH_PARAM = "reference_width_mm"
REFERENCE_HEIGHT_PARAM = "reference_height_mm"
REFERENCE_SIZE_PARAM = "reference_size_mm"
SCALE_PARAM = "scale"
MEASURER_PARAM = "measurer"
DEBUG_PARAM = "debug"
DEBUG_IMAGES_PARAM = "debug_images"
DEBUG_IMAGE_URLS_PARAM = "debug_image_urls"
DEBUG_IMAGE_FORMAT = ".png"
DEBUG_IMAGE_MIME_TYPE = "image/png"
SAVED_DEBUG_IMAGE_DEFAULT_MIME_TYPE = "application/octet-stream"
OBJECT_CONTOUR_SVG_KEY = "object_contour_svg"
DXF_UPLOAD_FIELD = "file"
DXF_ALTERNATE_UPLOAD_FIELD = "dxf"
DXF_SCALE_PARAM = "scale"
DXF_FIT_PAGE_PARAM = "fit_page"
DXF_MARGIN_PARAM = "margin"
DXF_PAGE_WIDTH_PARAM = "page_width"
DXF_PAGE_HEIGHT_PARAM = "page_height"
DXF_UNITS_PARAM = "units"
DXF_DEFAULT_SCALE = 1.0
DXF_DEFAULT_MARGIN = 0.0
DXF_DEFAULT_PAGE_SIZE = 0.0
DXF_DEFAULT_UNITS = "px"
DXF_OUTPUT_COORDINATE_SPACE = 1_000_000.0
DXF_PAGE_UNITS = {
    "px": "px",
    "mm": "mm",
    "cm": "cm",
    "in": "inch",
    "inch": "inch",
    "pt": "pt",
}
DXF_TEXT_ENTITY_TYPES = {"TEXT", "MTEXT", "ATTRIB", "ATTDEF"}
SVG_NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
TRUE_QUERY_VALUES = {"1", "true", "yes", "on"}
FALSE_QUERY_VALUES = {"0", "false", "no", "off"}
OBJECT_MEASURER_NAME = "object"
REFERENCE_SURFACE_MEASURER_NAME = "reference_surface"
EXAMPLE_IMAGE_URL = (
    "https://raw.githubusercontent.com/justingeeslin/Real-Time-Object-Measurement/"
    "main/test-images/ucard-one-off-axis/ucard-one-off-axis.jpg"
)


class BadReferenceSize(ValueError):
    pass


class BadMeasurementOption(ValueError):
    pass


class BadDxfConversionOption(ValueError):
    pass


class BadDxfUpload(ValueError):
    pass


class DxfConversionDependencyMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class DxfSvgOptions:
    scale: float = DXF_DEFAULT_SCALE
    fit_page: bool = False
    margin: float = DXF_DEFAULT_MARGIN
    page_width: float = DXF_DEFAULT_PAGE_SIZE
    page_height: float = DXF_DEFAULT_PAGE_SIZE
    units: str = DXF_DEFAULT_UNITS


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


def _parse_non_negative_float(value, param_name):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise BadDxfConversionOption(f"'{param_name}' must be a number.")

    if not math.isfinite(parsed) or parsed < 0:
        raise BadDxfConversionOption(
            f"'{param_name}' must be a finite number greater than or equal to 0."
        )

    return parsed


def _parse_positive_measurement_float(value, param_name):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise BadMeasurementOption(f"'{param_name}' must be a number.")

    if not math.isfinite(parsed) or parsed <= 0:
        raise BadMeasurementOption(
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


def get_measurement_kwargs(query_params):
    kwargs = {"reference_size_mm": get_reference_size_mm(query_params)}

    scale = query_params.get(SCALE_PARAM)
    if scale is not None:
        kwargs["scale"] = _parse_positive_measurement_float(scale, SCALE_PARAM)

    return kwargs


def get_measurer_name(query_params):
    value = query_params.get(MEASURER_PARAM, OBJECT_MEASURER_NAME)
    normalized = str(value).strip().lower().replace("-", "_")
    compact = normalized.replace("_", "")

    if compact in {"default", "legacy", "object", "objectmeasurer"}:
        return OBJECT_MEASURER_NAME

    if compact in {
        "reference",
        "referencesurface",
        "referencesurfacemeasurer",
    }:
        return REFERENCE_SURFACE_MEASURER_NAME

    raise BadMeasurementOption(
        "'measurer' must be either 'object' or 'reference_surface'."
    )


def get_measurer_class(measurer_name):
    if measurer_name == OBJECT_MEASURER_NAME:
        return ObjectMeasurer

    if ReferenceSurfaceMeasurer is None:
        raise BadMeasurementOption(
            "'measurer=reference_surface' requires ReferenceSurfaceMeasurer to be installed."
        )

    return ReferenceSurfaceMeasurer


def _query_param_enabled(query_params, param_name):
    value = query_params.get(param_name)
    if value is None:
        return False

    return str(value).strip().lower() in TRUE_QUERY_VALUES


def _request_param(request, param_name, default=None):
    if param_name in request.POST:
        return request.POST.get(param_name)

    return request.GET.get(param_name, default)


def _parse_dxf_positive_float_param(request, param_name, default):
    value = _request_param(request, param_name)
    if value is None:
        return default

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise BadDxfConversionOption(f"'{param_name}' must be a number.")

    if not math.isfinite(parsed) or parsed <= 0:
        raise BadDxfConversionOption(
            f"'{param_name}' must be a finite number greater than 0."
        )

    return parsed


def _parse_dxf_non_negative_float_param(request, param_name, default):
    value = _request_param(request, param_name)
    if value is None:
        return default

    return _parse_non_negative_float(value, param_name)


def _parse_dxf_bool_param(request, param_name, default):
    value = _request_param(request, param_name)
    if value is None:
        return default

    normalized = str(value).strip().lower()
    if normalized in TRUE_QUERY_VALUES:
        return True
    if normalized in FALSE_QUERY_VALUES:
        return False

    raise BadDxfConversionOption(
        f"'{param_name}' must be one of: "
        f"{', '.join(sorted(TRUE_QUERY_VALUES | FALSE_QUERY_VALUES))}."
    )


def get_dxf_svg_options(request):
    units = (
        str(_request_param(request, DXF_UNITS_PARAM, DXF_DEFAULT_UNITS))
        .strip()
        .lower()
    )
    if units not in DXF_PAGE_UNITS:
        raise BadDxfConversionOption(
            f"'{DXF_UNITS_PARAM}' must be one of: {', '.join(sorted(DXF_PAGE_UNITS))}."
        )

    return DxfSvgOptions(
        scale=_parse_dxf_positive_float_param(
            request,
            DXF_SCALE_PARAM,
            DXF_DEFAULT_SCALE,
        ),
        fit_page=_parse_dxf_bool_param(request, DXF_FIT_PAGE_PARAM, False),
        margin=_parse_dxf_non_negative_float_param(
            request,
            DXF_MARGIN_PARAM,
            DXF_DEFAULT_MARGIN,
        ),
        page_width=_parse_dxf_non_negative_float_param(
            request,
            DXF_PAGE_WIDTH_PARAM,
            DXF_DEFAULT_PAGE_SIZE,
        ),
        page_height=_parse_dxf_non_negative_float_param(
            request,
            DXF_PAGE_HEIGHT_PARAM,
            DXF_DEFAULT_PAGE_SIZE,
        ),
        units=units,
    )


def get_debug_image_root():
    return Path(settings.DEBUG_IMAGE_ROOT)


def build_debug_image_request_path():
    return get_debug_image_root() / uuid.uuid4().hex


def measurer_supports_saved_debug_images(measurer_class):
    try:
        parameters = inspect.signature(measurer_class).parameters
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
                "name": str(image.get("name", image.get("label", ""))),
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


def _format_svg_number(value):
    return f"{float(value):g}"


def _debug_image_viewbox(debug):
    image = debug.get("imgWarp")
    if not _is_debug_image(image):
        return None

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return None

    return (0.0, 0.0, float(width), float(height))


def _path_data_viewbox(path_data):
    values = [
        float(match.group(0))
        for match in SVG_NUMBER_PATTERN.finditer(path_data)
    ]
    points = list(zip(values[0::2], values[1::2]))
    if not points:
        return (0.0, 0.0, 1.0, 1.0)

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    min_x = min(x_values)
    min_y = min(y_values)
    width = max(x_values) - min_x
    height = max(y_values) - min_y

    if width <= 0:
        width = 1.0
    if height <= 0:
        height = 1.0

    return (min_x, min_y, width, height)


def build_object_contour_svg(value, debug):
    if not isinstance(value, str):
        return value

    contour = value.strip()
    if not contour:
        return value

    if contour.lstrip().lower().startswith("<svg"):
        return value

    min_x, min_y, width, height = _path_data_viewbox(contour)
    viewbox = " ".join(
        _format_svg_number(number) for number in (min_x, min_y, width, height)
    )
    escaped_path = html.escape(contour, quote=True)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}">'
        f'<path d="{escaped_path}" fill="none" stroke="currentColor" '
        f'stroke-width="2" vector-effect="non-scaling-stroke"/></svg>'
    )


def normalize_debug_for_response(debug):
    if OBJECT_CONTOUR_SVG_KEY not in debug:
        return debug

    normalized = dict(debug)
    normalized[OBJECT_CONTOUR_SVG_KEY] = build_object_contour_svg(
        debug[OBJECT_CONTOUR_SVG_KEY],
        debug,
    )
    return normalized


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

    if OBJECT_CONTOUR_SVG_KEY in debug:
        data["svg"] = debug[OBJECT_CONTOUR_SVG_KEY]

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
        "height": measurement.height_mm,
        "width": measurement.width_mm,
    }

    bbox = getattr(measurement, "bbox", None)
    if bbox is not None:
        serialized["bbox"] = [int(value) for value in bbox]

    return serialized


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
            "Use ReferenceSurfaceMeasurer",
            "Measures with the alternate reference-surface implementation.",
            {
                "url": EXAMPLE_IMAGE_URL,
                MEASURER_PARAM: REFERENCE_SURFACE_MEASURER_NAME,
            },
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


def _load_ezdxf_drawing_modules():
    try:
        import ezdxf
        from ezdxf import recover
        from ezdxf.addons.drawing import Frontend, RenderContext, config, layout, svg
    except ImportError as exc:
        missing_dependency = getattr(exc, "name", None) or str(exc)
        raise DxfConversionDependencyMissing(
            "DXF to SVG conversion requires the ezdxf drawing dependencies. "
            f"Missing import: {missing_dependency}."
        ) from exc

    return ezdxf, recover, Frontend, RenderContext, config, layout, svg


def _read_dxf_document(dxf_path, ezdxf, recover):
    try:
        return ezdxf.readfile(dxf_path)
    except (IOError, UnicodeDecodeError, ezdxf.DXFStructureError) as first_error:
        try:
            doc, auditor = recover.readfile(dxf_path)
        except (IOError, UnicodeDecodeError, ezdxf.DXFStructureError) as exc:
            raise BadDxfUpload(
                "Uploaded file is not a readable DXF document."
            ) from exc

        if auditor.has_errors:
            raise BadDxfUpload(
                "Uploaded DXF contains unrecoverable structure errors."
            ) from first_error

        return doc


def _iter_unique_dxf_entity_spaces(doc):
    seen = set()

    for entity_spaces in (doc.layouts, doc.blocks):
        for entity_space in entity_spaces:
            raw_entity_space = getattr(entity_space, "entity_space", entity_space)
            marker = id(raw_entity_space)
            if marker in seen:
                continue

            seen.add(marker)
            yield entity_space


def _remove_dxf_text_entities(doc):
    for entity_space in _iter_unique_dxf_entity_spaces(doc):
        for entity in list(entity_space):
            entity_type = entity.dxftype()

            if entity_type == "INSERT" and getattr(entity, "attribs", None):
                entity.delete_all_attribs()

            if entity_type in DXF_TEXT_ENTITY_TYPES:
                entity_space.delete_entity(entity)


def render_dxf_path_to_svg(dxf_path, options):
    ezdxf, recover, Frontend, RenderContext, config, layout, svg = (
        _load_ezdxf_drawing_modules()
    )
    doc = _read_dxf_document(dxf_path, ezdxf, recover)
    _remove_dxf_text_entities(doc)
    page_units = getattr(layout.Units, DXF_PAGE_UNITS[options.units])

    context = RenderContext(doc)
    backend = svg.SVGBackend()
    render_config = config.Configuration(
        background_policy=config.BackgroundPolicy.WHITE,
        color_policy=config.ColorPolicy.BLACK,
    )
    Frontend(context, backend, config=render_config).draw_layout(doc.modelspace())

    page = layout.Page(
        options.page_width,
        options.page_height,
        page_units,
        margins=layout.Margins.all(options.margin),
    )
    settings = layout.Settings(
        scale=options.scale,
        fit_page=options.fit_page,
        output_coordinate_space=DXF_OUTPUT_COORDINATE_SPACE * options.scale,
    )
    return backend.get_string(page, settings=settings)


def convert_uploaded_dxf_to_svg(uploaded_file, options):
    dxf_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as dxf_file:
            dxf_path = Path(dxf_file.name)
            for chunk in uploaded_file.chunks():
                dxf_file.write(chunk)

        return render_dxf_path_to_svg(dxf_path, options)
    finally:
        if dxf_path is not None:
            dxf_path.unlink(missing_ok=True)


def _svg_download_name(uploaded_file):
    stem = Path(uploaded_file.name or "drawing").stem
    safe_stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in stem
    ).strip("_")

    return f"{safe_stem or 'drawing'}.svg"


@csrf_exempt
@require_POST
def dxf_to_svg(request):
    uploaded_file = request.FILES.get(DXF_UPLOAD_FIELD) or request.FILES.get(
        DXF_ALTERNATE_UPLOAD_FIELD
    )
    if uploaded_file is None:
        return JsonResponse(
            {
                "error": "Missing DXF upload",
                "details": (
                    f"Upload a multipart file field named '{DXF_UPLOAD_FIELD}' "
                    f"or '{DXF_ALTERNATE_UPLOAD_FIELD}'."
                ),
            },
            status=400,
        )

    try:
        options = get_dxf_svg_options(request)
        svg_markup = convert_uploaded_dxf_to_svg(uploaded_file, options)
    except BadDxfConversionOption as exc:
        return JsonResponse(
            {"error": "Invalid DXF conversion option", "details": str(exc)},
            status=400,
        )
    except BadDxfUpload as exc:
        return JsonResponse(
            {"error": "Invalid DXF upload", "details": str(exc)},
            status=400,
        )
    except DxfConversionDependencyMissing as exc:
        return JsonResponse(
            {"error": "DXF conversion unavailable", "details": str(exc)},
            status=503,
        )
    except Exception as exc:
        logger.exception("DXF to SVG conversion failed")
        return JsonResponse(
            {"error": "DXF to SVG conversion failed", "details": str(exc)},
            status=500,
        )

    response = HttpResponse(svg_markup, content_type="image/svg+xml; charset=utf-8")
    response["Content-Disposition"] = (
        f'inline; filename="{_svg_download_name(uploaded_file)}"'
    )
    response["X-DXF-SVG-Scale"] = str(options.scale)
    response["X-DXF-SVG-Units"] = options.units
    return response


def measure(request):
    image_url = request.GET.get("url")
    wants_debug_image_urls = _query_param_enabled(request.GET, DEBUG_IMAGE_URLS_PARAM)

    if not image_url:
        return index(request)

    try:
        measurer_name = get_measurer_name(request.GET)
        measurer_class = get_measurer_class(measurer_name)
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
        and measurer_supports_saved_debug_images(measurer_class)
    ):
        measurement_kwargs["debug_path"] = str(debug_image_request_path)
        measurement_kwargs["save_debug_images"] = True

    # if measurer_class == "object":
    #     measurer = ObjectMeasurer(**measurement_kwargs)
    # else:
    measurer = ReferenceSurfaceMeasurer(**measurement_kwargs)

    data = {"measurer": "ReferenceSurfaceMeasurer"}

    try:
        # Get the measurements (in mm)
        measurements, debug = measurer.measure(img, return_debug=True)

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

    width_mm, height_mm = measurements[0].width_mm, measurements[0].height_mm

    data["svg"] = debug[OBJECT_CONTOUR_SVG_KEY]
    data["height_mm"] = height_mm
    data["width_mm"] = width_mm
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
