import os
import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import urljoin
from xml.etree import ElementTree as ET
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

import pytest
import requests


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "gettingstarted.settings")

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES_ROOT = REPO_ROOT / "test-images"

A4_MM = (210.0, 297.0)
LETTER_MM = (215.9, 279.4)  # 8.5in x 11in
PORTRAIT_POSTER_BOARD_MM = (561.975, 711.2)
ODDBALL_BLACK_POSTER_BOARD_MM = (508, 752.475)
LIGHTBOX_MAT_MM = (746.125, 1098.55)

MEASURERS = ["object", "reference_surface"]
SVG_NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
SVG_SHAPE_RELATIVE_TOLERANCE = 0.25
MEASUREMENT_RELATIVE_TOLERANCE = 0.25
ENDPOINT_PASS_CRITERIA = (
    "status=200, echoed measurer, debug.status=ok, measurements present, "
    "width>0, height>0"
)
COMPARISON_PASS_CRITERIA = (
    "top-level keys identical, first measurement keys identical, "
    f"physical dimensions within {MEASUREMENT_RELATIVE_TOLERANCE:.0%}, "
    f"normalized SVG shape within {SVG_SHAPE_RELATIVE_TOLERANCE:.0%}"
)
IMAGE_CASES = [
    ("ucard-one", "ucard-one.jpg", LETTER_MM, 1),
    ("iswic-folded", "img_6a1db6e03ad2a7.16505140.jpg", PORTRAIT_POSTER_BOARD_MM, 1),
    ("iswic-folded2", "img_6a1db77d7f1973.02751931.jpg", PORTRAIT_POSTER_BOARD_MM, 1),
    ("goldy-lightblue", "img_6a1db5fd6d4312.38941970.jpg", PORTRAIT_POSTER_BOARD_MM, 1),
    ("one", "one.jpg", A4_MM, 1),
    ("ucard-two", "ucard-two.jpg", LETTER_MM, 1),
    ("ucard-one-off-axis", "ucard-one-off-axis.jpg", LETTER_MM, 1),
    ("ucard-two-off-axis", "ucard-two-off-axis.jpg", LETTER_MM, 1),
    ("nike-letter-one", "img_6a7c9d736092f6.97385818.jpg", LETTER_MM, 1),
    ("nike-letter-one-off-axis", "img_6a838cf9edde22.59423143.jpg", LETTER_MM, 1),
    ("iswic", "iswic.jpg", LIGHTBOX_MAT_MM, 1),
    ("goldy", "goldy.jpg", ODDBALL_BLACK_POSTER_BOARD_MM, 1),
    ("cherokee", "cherokee.jpg", ODDBALL_BLACK_POSTER_BOARD_MM, 1),
]
COMPARISON_IMAGE_CASES = [
    ("ucard-one", "ucard-one.jpg", LETTER_MM, 1),
    ("ucard-two", "ucard-two.jpg", LETTER_MM, 1),
    ("ucard-one-off-axis", "ucard-one-off-axis.jpg", LETTER_MM, 1),
    ("ucard-two-off-axis", "ucard-two-off-axis.jpg", LETTER_MM, 1),
    ("iswic-folded", "img_6a1db6e03ad2a7.16505140.jpg", PORTRAIT_POSTER_BOARD_MM, 1),
    ("iswic-folded2", "img_6a1db77d7f1973.02751931.jpg", PORTRAIT_POSTER_BOARD_MM, 1),
    ("goldy", "goldy.jpg", ODDBALL_BLACK_POSTER_BOARD_MM, 1),
    ("cherokee", "cherokee.jpg", ODDBALL_BLACK_POSTER_BOARD_MM, 1),
]


class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


class QuietWSGIRequestHandler(WSGIRequestHandler):
    def log_message(self, format, *args):
        pass


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


def _server_url(server):
    host, port = server.server_address[:2]
    return f"http://{host}:{port}/"


def _start_server(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


@pytest.fixture(scope="session")
def test_image_base_url():
    handler = partial(QuietHTTPRequestHandler, directory=str(TEST_IMAGES_ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = _start_server(server)

    try:
        yield _server_url(server)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="session")
def endpoint_url():
    from django.core.wsgi import get_wsgi_application

    server = make_server(
        "127.0.0.1",
        0,
        get_wsgi_application(),
        server_class=ThreadingWSGIServer,
        handler_class=QuietWSGIRequestHandler,
    )
    thread = _start_server(server)

    try:
        yield _server_url(server)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _measurement_params(
    slug, filename, reference_size_mm, scale, measurer, test_image_base_url
):
    image_url = urljoin(test_image_base_url, f"{slug}/{filename}")
    return {
        "url": image_url,
        "measurer": measurer,
        "reference_size_mm": ",".join(map(str, reference_size_mm)),
        "scale": scale,
        "debug": 1,
    }


def _fetch_measurement_payload(
    slug,
    filename,
    reference_size_mm,
    scale,
    measurer,
    endpoint_url,
    test_image_base_url,
):
    response = requests.get(
        endpoint_url,
        params=_measurement_params(
            slug,
            filename,
            reference_size_mm,
            scale,
            measurer,
            test_image_base_url,
        ),
        timeout=120,
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["measurer"] == measurer
    assert payload["debug"]["status"] == "ok", payload
    assert payload["measurements"], payload
    assert payload["width"] > 0
    assert payload["height"] > 0
    return payload


def _svg_markup(svg_value):
    if isinstance(svg_value, (list, tuple)):
        assert svg_value, "SVG payload list is empty"
        return svg_value[0]

    return svg_value


def _svg_numbers(value):
    return [float(match.group(0)) for match in SVG_NUMBER_PATTERN.finditer(value)]


def _coordinate_size_from_values(values):
    points = list(zip(values[0::2], values[1::2]))
    assert points, "SVG payload did not contain any coordinate pairs"

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return max(x_values) - min(x_values), max(y_values) - min(y_values)


def _svg_viewbox_size(root):
    viewbox = root.attrib.get("viewBox")
    if viewbox:
        values = _svg_numbers(viewbox)
        assert len(values) == 4, f"Unexpected SVG viewBox: {viewbox}"
        return values[2], values[3]

    width = root.attrib.get("width")
    height = root.attrib.get("height")
    if width and height:
        return _svg_numbers(width)[0], _svg_numbers(height)[0]

    return None


def _svg_content_size(svg_value):
    markup = _svg_markup(svg_value)
    assert isinstance(markup, str), f"Unexpected SVG payload: {svg_value!r}"

    stripped = markup.strip()
    if stripped.startswith("<"):
        root = ET.fromstring(stripped)
        coordinate_attributes = []
        for element in root.iter():
            coordinate_attributes.extend(
                value
                for key, value in element.attrib.items()
                if key in {"d", "points"}
            )

        if coordinate_attributes:
            return _coordinate_size_from_values(
                _svg_numbers(" ".join(coordinate_attributes))
            )

        viewbox_size = _svg_viewbox_size(root)
        assert viewbox_size is not None, "SVG payload did not contain path or size data"
        return viewbox_size

    return _coordinate_size_from_values(_svg_numbers(markup))


def _normal_size(size):
    width, height = size
    assert width > 0
    assert height > 0
    longest = max(width, height)
    return sorted((width / longest, height / longest))


def _assert_sizes_close(actual_size, expected_size, relative_tolerance):
    actual = sorted(actual_size)
    expected = sorted(expected_size)

    for actual_dimension, expected_dimension in zip(actual, expected):
        assert actual_dimension == pytest.approx(
            expected_dimension,
            rel=relative_tolerance,
        )


def _image_id(slug, filename):
    return f"{slug}/{filename}"


def _fmt_number(value):
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _fmt_size(size):
    width, height = size
    return f"{_fmt_number(width)} x {_fmt_number(height)}"


def _endpoint_summary(slug, filename, reference_size_mm, scale, measurer, payload):
    measurement_count = len(payload["measurements"])
    return (
        f"[endpoint] image={_image_id(slug, filename)} measurer={measurer} "
        f"reference_mm={_fmt_size(reference_size_mm)} scale={scale} "
        f"returned_cm={_fmt_size((payload['width'], payload['height']))} "
        f"measurements={measurement_count}; pass={ENDPOINT_PASS_CRITERIA}"
    )


def _comparison_summary(
    slug,
    filename,
    object_payload,
    reference_payload,
    object_svg_size,
    reference_svg_size,
):
    return (
        f"[comparison] image={_image_id(slug, filename)} "
        f"object_cm={_fmt_size((object_payload['width'], object_payload['height']))} "
        f"reference_surface_cm="
        f"{_fmt_size((reference_payload['width'], reference_payload['height']))} "
        f"object_svg_norm={_fmt_size(object_svg_size)} "
        f"reference_surface_svg_norm={_fmt_size(reference_svg_size)}; "
        f"pass={COMPARISON_PASS_CRITERIA}"
    )


def _print_live(capsys, message):
    with capsys.disabled():
        print(message, flush=True)


@pytest.mark.parametrize("measurer", MEASURERS)
@pytest.mark.parametrize(
    "slug,filename,reference_size_mm,scale",
    IMAGE_CASES,
    ids=[case[0] for case in IMAGE_CASES],
)
def test_measurement_endpoint(
    slug,
    filename,
    reference_size_mm,
    scale,
    measurer,
    endpoint_url,
    test_image_base_url,
    capsys,
):
    payload = _fetch_measurement_payload(
        slug,
        filename,
        reference_size_mm,
        scale,
        measurer,
        endpoint_url,
        test_image_base_url,
    )
    _print_live(
        capsys,
        _endpoint_summary(slug, filename, reference_size_mm, scale, measurer, payload),
    )


@pytest.mark.parametrize(
    "slug,filename,reference_size_mm,scale",
    COMPARISON_IMAGE_CASES,
    ids=[case[0] for case in COMPARISON_IMAGE_CASES],
)
def test_measurement_endpoint_measurer_responses_match_shape(
    slug,
    filename,
    reference_size_mm,
    scale,
    endpoint_url,
    test_image_base_url,
    capsys,
):
    payloads = {
        measurer: _fetch_measurement_payload(
            slug,
            filename,
            reference_size_mm,
            scale,
            measurer,
            endpoint_url,
            test_image_base_url,
        )
        for measurer in MEASURERS
    }

    object_payload = payloads["object"]
    reference_payload = payloads["reference_surface"]

    assert set(object_payload) == set(reference_payload)
    assert set(object_payload["measurements"][0]) == set(
        reference_payload["measurements"][0]
    )

    _assert_sizes_close(
        (object_payload["width"], object_payload["height"]),
        (reference_payload["width"], reference_payload["height"]),
        MEASUREMENT_RELATIVE_TOLERANCE,
    )
    object_svg_size = _normal_size(_svg_content_size(object_payload["svg"]))
    reference_svg_size = _normal_size(_svg_content_size(reference_payload["svg"]))
    _assert_sizes_close(
        object_svg_size,
        reference_svg_size,
        SVG_SHAPE_RELATIVE_TOLERANCE,
    )
    _print_live(
        capsys,
        _comparison_summary(
            slug,
            filename,
            object_payload,
            reference_payload,
            object_svg_size,
            reference_svg_size,
        ),
    )
