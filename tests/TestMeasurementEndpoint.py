import os
from urllib.parse import urljoin

import pytest
import requests


A4_MM = (210.0, 297.0)
LETTER_MM = (215.9, 279.4)  # 8.5in x 11in
PORTRAIT_POSTER_BOARD_MM = (561.975, 711.2)
ODDBALL_BLACK_POSTER_BOARD_MM = (508, 752.475)
DOUBLE_STACKED_LANDSCAPE_POSTER_BOARD_MM = (711.2, 1123.95)
MOCK_BOX_MM = (762, 755.65)
SQ_CANVAS_MM = (762, 762)
LIGHTBOX_MAT_MM = (746.125, 1098.55)
ENVELOPE_MM = (184, 133)

ENDPOINT_URL = os.getenv("MEASUREMENT_ENDPOINT_URL")
TEST_IMAGE_BASE_URL = os.getenv(
    "TEST_IMAGE_BASE_URL",
    "https://phpstack-1496460-6260926.cloudwaysapps.com/test-images/",
)

MEASURERS = ["object", "reference_surface"]
IMAGE_CASES = [
    ("ucard-one", "ucard-one.jpg", LETTER_MM, 1),
    ("iswic-folded", "img_6a1db6e03ad2a7.16505140.jpg", PORTRAIT_POSTER_BOARD_MM, 1),
    ("iswic-folded2", "img_6a1db77d7f1973.02751931.jpg", PORTRAIT_POSTER_BOARD_MM, 1),
    ("goldy-lightblue", "img_6a1db5fd6d4312.38941970.jpg", PORTRAIT_POSTER_BOARD_MM, 1),
    ("mock-box-white-blue-one", "IMG_0315.jpeg", MOCK_BOX_MM, 1),
    ("mock-box-white-blue-goldy-one-off-axis", "img_6a84aaa9bd5684.29594166.jpg", MOCK_BOX_MM, 1),
    ("mock-box-white-blue-goldy-one", "IMG_0317.jpeg", MOCK_BOX_MM, 1),
    ("mock-box-white-blue-goldy-two", "img_6a84a94c724176.66225541.jpg", MOCK_BOX_MM, 1),
    ("mock-box-white-blue-goldy-three", "img_6a84a977ae3347.44496591.jpg", MOCK_BOX_MM, 1),
    ("mock-box-white-blue-goldy-four", "img_6a84a8e7b20297.13560456.jpg", MOCK_BOX_MM, 1),
    ("one", "one.jpg", A4_MM, 1),
    ("ucard-two", "ucard-two.jpg", LETTER_MM, 1),
    ("ucard-one-off-axis", "ucard-one-off-axis.jpg", LETTER_MM, 1),
    ("ucard-two-off-axis", "ucard-two-off-axis.jpg", LETTER_MM, 1),
    ("nike-letter-one", "img_6a7c9d736092f6.97385818.jpg", LETTER_MM, 1),
    ("nike-letter-one-off-axis", "img_6a838cf9edde22.59423143.jpg", LETTER_MM, 1),
    ("iswic", "iswic.jpg", LIGHTBOX_MAT_MM, 1),
    ("goldy", "goldy.jpg", ODDBALL_BLACK_POSTER_BOARD_MM, 1),
    ("cherokee", "cherokee.jpg", ODDBALL_BLACK_POSTER_BOARD_MM, 1),
    ("iswic-30x30-canvas", "iswic-30x30-canvas.jpg", SQ_CANVAS_MM, 1),
]

pytestmark = pytest.mark.skipif(
    not ENDPOINT_URL,
    reason="Set MEASUREMENT_ENDPOINT_URL to run endpoint end-to-end tests",
)


@pytest.mark.parametrize("measurer", MEASURERS)
@pytest.mark.parametrize(
    "slug,filename,reference_size_mm,scale",
    IMAGE_CASES,
    ids=[case[0] for case in IMAGE_CASES],
)
def test_measurement_endpoint(slug, filename, reference_size_mm, scale, measurer):
    image_url = urljoin(TEST_IMAGE_BASE_URL, f"{slug}/{filename}")
    response = requests.get(
        ENDPOINT_URL,
        params={
            "url": image_url,
            "measurer": measurer,
            "reference_size_mm": ",".join(map(str, reference_size_mm)),
            "scale": scale,
            "debug": 1,
        },
        timeout=120,
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["measurer"] == measurer
    assert payload["debug"]["status"] == "ok", payload
    assert payload["measurements"], payload
    assert payload["width"] > 0
    assert payload["height"] > 0
