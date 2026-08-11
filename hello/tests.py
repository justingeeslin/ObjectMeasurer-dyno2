import base64
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from django.test import SimpleTestCase

from .views import PORTRAIT_POSTER_BOARD_MM


class MeasureEndpointTest(SimpleTestCase):
    def configure_successful_measurement(self, requests_get, imdecode, object_measurer):
        requests_get.return_value = SimpleNamespace(
            ok=True,
            content=b"image-bytes",
            headers={},
            status_code=200,
            text="",
        )
        imdecode.return_value = object()

        measurer = object_measurer.return_value
        measurer.debug = {}
        measurer.measure.return_value = [
            SimpleNamespace(width_cm=12.3, height_cm=45.6),
        ]
        return measurer

    @patch("hello.views.ObjectMeasurer")
    @patch("hello.views.cv2.imdecode")
    @patch("hello.views.requests.get")
    def test_measure_uses_default_reference_size(
        self, requests_get, imdecode, object_measurer
    ):
        self.configure_successful_measurement(requests_get, imdecode, object_measurer)

        response = self.client.get("/", {"url": "https://example.com/photo.jpg"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"height": 45.6, "width": 12.3})
        object_measurer.assert_called_once_with(
            reference_size_mm=PORTRAIT_POSTER_BOARD_MM
        )

    @patch("hello.views.ObjectMeasurer")
    @patch("hello.views.cv2.imdecode")
    @patch("hello.views.requests.get")
    def test_measure_returns_base64_debug_images_when_requested(
        self, requests_get, imdecode, object_measurer
    ):
        measurer = self.configure_successful_measurement(
            requests_get, imdecode, object_measurer
        )
        measurer.debug = {
            "imgWarp": np.zeros((2, 3, 3), dtype=np.uint8),
            "not_an_image": "debug text",
        }

        response = self.client.get(
            "/",
            {
                "url": "https://example.com/photo.jpg",
                "debug_images": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        debug_images = response.json()["debug_images"]
        self.assertEqual(set(debug_images), {"imgWarp"})

        image_payload = debug_images["imgWarp"]
        self.assertEqual(image_payload["mime_type"], "image/png")
        self.assertEqual(image_payload["encoding"], "base64")
        self.assertEqual(image_payload["width"], 3)
        self.assertEqual(image_payload["height"], 2)
        self.assertEqual(image_payload["shape"], [2, 3, 3])
        self.assertEqual(image_payload["dtype"], "uint8")
        self.assertEqual(
            base64.b64decode(image_payload["data"])[:8],
            b"\x89PNG\r\n\x1a\n",
        )

    @patch("hello.views.ObjectMeasurer")
    @patch("hello.views.cv2.imdecode")
    @patch("hello.views.requests.get")
    def test_measure_accepts_reference_width_and_height_mm(
        self, requests_get, imdecode, object_measurer
    ):
        self.configure_successful_measurement(requests_get, imdecode, object_measurer)

        response = self.client.get(
            "/",
            {
                "url": "https://example.com/photo.jpg",
                "reference_width_mm": "100.5",
                "reference_height_mm": "200.25",
            },
        )

        self.assertEqual(response.status_code, 200)
        object_measurer.assert_called_once_with(reference_size_mm=(100.5, 200.25))

    @patch("hello.views.ObjectMeasurer")
    @patch("hello.views.cv2.imdecode")
    @patch("hello.views.requests.get")
    def test_measure_accepts_compact_reference_size_mm(
        self, requests_get, imdecode, object_measurer
    ):
        self.configure_successful_measurement(requests_get, imdecode, object_measurer)

        response = self.client.get(
            "/",
            {
                "url": "https://example.com/photo.jpg",
                "reference_size_mm": "215.9,279.4",
            },
        )

        self.assertEqual(response.status_code, 200)
        object_measurer.assert_called_once_with(reference_size_mm=(215.9, 279.4))

    @patch("hello.views.requests.get")
    def test_measure_rejects_partial_reference_dimensions(self, requests_get):
        response = self.client.get(
            "/",
            {
                "url": "https://example.com/photo.jpg",
                "reference_width_mm": "100",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["error"],
            "Invalid reference object dimensions",
        )
        self.assertIn("provided together", response.json()["details"])
        requests_get.assert_not_called()

    @patch("hello.views.requests.get")
    def test_measure_rejects_non_positive_reference_dimensions(self, requests_get):
        response = self.client.get(
            "/",
            {
                "url": "https://example.com/photo.jpg",
                "reference_size_mm": "0,279.4",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["error"],
            "Invalid reference object dimensions",
        )
        self.assertIn("greater than 0", response.json()["details"])
        requests_get.assert_not_called()

    @patch("hello.views.requests.get")
    def test_measure_rejects_non_finite_reference_dimensions(self, requests_get):
        response = self.client.get(
            "/",
            {
                "url": "https://example.com/photo.jpg",
                "reference_width_mm": "nan",
                "reference_height_mm": "279.4",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["error"],
            "Invalid reference object dimensions",
        )
        self.assertIn("finite number", response.json()["details"])
        requests_get.assert_not_called()
