import base64
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlparse

import cv2
import numpy as np
from django.test import SimpleTestCase, override_settings

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

    @patch("hello.views.requests.get")
    def test_root_without_url_shows_index_with_examples(self, requests_get):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "text/html; charset=utf-8")

        content = response.content.decode()
        self.assertIn("Object Measurement API", content)
        self.assertIn("Sample Links", content)
        self.assertIn("reference_size_mm=215.9%2C279.4", content)
        self.assertIn("measurer=reference_surface", content)
        self.assertIn("debug=1", content)
        self.assertIn("debug_images=1", content)
        self.assertIn("debug_image_urls=1", content)
        self.assertIn("data:image/png;base64,&lt;data&gt;", content)
        requests_get.assert_not_called()

    @patch("hello.views.ObjectMeasurer")
    @patch("hello.views.cv2.imdecode")
    @patch("hello.views.requests.get")
    def test_measure_uses_default_reference_size(
        self, requests_get, imdecode, object_measurer
    ):
        self.configure_successful_measurement(requests_get, imdecode, object_measurer)

        response = self.client.get("/", {"url": "https://example.com/photo.jpg"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["height"], 45.6)
        self.assertEqual(response.json()["width"], 12.3)
        self.assertEqual(
            response.json()["measurements"],
            [{"height": 45.6, "width": 12.3}],
        )
        object_measurer.assert_called_once_with(
            reference_size_mm=PORTRAIT_POSTER_BOARD_MM
        )

    @patch("hello.views.ReferenceSurfaceMeasurer")
    @patch("hello.views.ObjectMeasurer")
    @patch("hello.views.cv2.imdecode")
    @patch("hello.views.requests.get")
    def test_measure_accepts_reference_surface_measurer(
        self,
        requests_get,
        imdecode,
        object_measurer,
        reference_surface_measurer,
    ):
        self.configure_successful_measurement(
            requests_get,
            imdecode,
            reference_surface_measurer,
        )

        response = self.client.get(
            "/",
            {
                "url": "https://example.com/photo.jpg",
                "measurer": "reference_surface",
                "reference_size_mm": "215.9,279.4",
                "scale": "2",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["measurer"], "reference_surface")
        object_measurer.assert_not_called()
        reference_surface_measurer.assert_called_once_with(
            reference_size_mm=(215.9, 279.4),
            scale=2,
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
        self.assertEqual(response.json()["debug"]["not_an_image"], "debug text")

    @patch("hello.views.uuid.uuid4")
    @patch("hello.views.ObjectMeasurer")
    @patch("hello.views.cv2.imdecode")
    @patch("hello.views.requests.get")
    def test_measure_returns_saved_debug_image_urls_when_requested(
        self, requests_get, imdecode, object_measurer, uuid4
    ):
        measurer = self.configure_successful_measurement(
            requests_get, imdecode, object_measurer
        )
        uuid4.return_value = SimpleNamespace(hex="debug-token")

        with tempfile.TemporaryDirectory() as debug_root:
            debug_dir = Path(debug_root) / "debug-token"
            debug_dir.mkdir()
            image_path = debug_dir / "0_untitled_warped.jpg"
            image_path.write_bytes(b"debug-image-bytes")
            measurer.debug = {
                "status": "ok",
                "debug_images": [
                    {
                        "name": "warped",
                        "path": str(image_path),
                        "saved": True,
                    },
                    {
                        "name": "missing",
                        "path": str(debug_dir / "missing.jpg"),
                        "saved": False,
                    },
                ],
            }

            with override_settings(DEBUG_IMAGE_ROOT=debug_root):
                response = self.client.get(
                    "/",
                    {
                        "url": "https://example.com/photo.jpg",
                        "debug_image_urls": "1",
                    },
                )

                self.assertEqual(response.status_code, 200)
                object_measurer.assert_called_once_with(
                    reference_size_mm=PORTRAIT_POSTER_BOARD_MM,
                    debug_path=str(debug_dir),
                    save_debug_images=True,
                )

                payload = response.json()
                self.assertEqual(payload["debug"]["status"], "ok")
                self.assertNotIn("debug_images", payload["debug"])

                debug_image_urls = payload["debug_image_urls"]
                self.assertEqual(len(debug_image_urls), 1)
                self.assertEqual(
                    debug_image_urls[0],
                    {
                        "name": "warped",
                        "filename": "0_untitled_warped.jpg",
                        "mime_type": "image/jpeg",
                        "url": (
                            "http://testserver/debug-images/debug-token/"
                            "0_untitled_warped.jpg"
                        ),
                    },
                )

                image_response = self.client.get(
                    urlparse(debug_image_urls[0]["url"]).path
                )
                self.assertEqual(image_response.status_code, 200)
                self.assertEqual(image_response.headers["Content-Type"], "image/jpeg")
                self.assertEqual(
                    b"".join(image_response.streaming_content),
                    b"debug-image-bytes",
                )

    @patch("hello.views.measurer_supports_saved_debug_images")
    @patch("hello.views.uuid.uuid4")
    @patch("hello.views.ObjectMeasurer")
    @patch("hello.views.cv2.imdecode")
    @patch("hello.views.requests.get")
    def test_measure_saves_debug_image_urls_when_measurer_has_no_save_hook(
        self, requests_get, imdecode, object_measurer, uuid4, supports_saved_images
    ):
        measurer = self.configure_successful_measurement(
            requests_get, imdecode, object_measurer
        )
        uuid4.return_value = SimpleNamespace(hex="fallback-token")
        supports_saved_images.return_value = False
        measurer.debug = {
            "status": "ok",
            "imgWarp": np.zeros((2, 3, 3), dtype=np.uint8),
        }

        with tempfile.TemporaryDirectory() as debug_root:
            with override_settings(DEBUG_IMAGE_ROOT=debug_root):
                response = self.client.get(
                    "/",
                    {
                        "url": "https://example.com/photo.jpg",
                        "debug_image_urls": "1",
                    },
                )

                self.assertEqual(response.status_code, 200)
                object_measurer.assert_called_once_with(
                    reference_size_mm=PORTRAIT_POSTER_BOARD_MM,
                )

                debug_image_urls = response.json()["debug_image_urls"]
                self.assertEqual(len(debug_image_urls), 1)
                self.assertEqual(debug_image_urls[0]["name"], "imgWarp")
                self.assertEqual(debug_image_urls[0]["filename"], "0_imgWarp.png")
                self.assertEqual(debug_image_urls[0]["mime_type"], "image/png")
                self.assertEqual(
                    debug_image_urls[0]["url"],
                    "http://testserver/debug-images/fallback-token/0_imgWarp.png",
                )

                image_response = self.client.get(
                    urlparse(debug_image_urls[0]["url"]).path
                )
                self.assertEqual(image_response.status_code, 200)
                self.assertEqual(image_response.headers["Content-Type"], "image/png")
                self.assertEqual(
                    b"".join(image_response.streaming_content)[:8],
                    b"\x89PNG\r\n\x1a\n",
                )

    @patch("hello.views.ObjectMeasurer")
    @patch("hello.views.cv2.imdecode")
    @patch("hello.views.requests.get")
    def test_measure_returns_serialized_debug_when_requested(
        self, requests_get, imdecode, object_measurer
    ):
        measurer = self.configure_successful_measurement(
            requests_get, imdecode, object_measurer
        )
        measurer.debug = {
            "status": "ok",
            "trace": [
                {
                    "event": "measure_start",
                    "reference_size_mm": (np.float64(215.9), np.float64(279.4)),
                }
            ],
            "page_detection": {
                "area": np.float64(123.4),
                "bbox": (1, 2, 3, 4),
            },
            "imgWarp": np.zeros((2, 3, 3), dtype=np.uint8),
            "invalid_score": float("nan"),
        }

        response = self.client.get(
            "/",
            {
                "url": "https://example.com/photo.jpg",
                "debug": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertNotIn("debug_images", payload)

        debug = payload["debug"]
        self.assertEqual(debug["status"], "ok")
        self.assertEqual(
            debug["trace"][0]["reference_size_mm"],
            [215.9, 279.4],
        )
        self.assertEqual(debug["page_detection"]["area"], 123.4)
        self.assertEqual(debug["page_detection"]["bbox"], [1, 2, 3, 4])
        self.assertEqual(
            debug["imgWarp"],
            {
                "type": "image",
                "shape": [2, 3, 3],
                "dtype": "uint8",
                "width": 3,
                "height": 2,
            },
        )
        self.assertNotIn("data", debug["imgWarp"])
        self.assertEqual(debug["invalid_score"], "nan")

    @patch("hello.views.ObjectMeasurer")
    @patch("hello.views.cv2.imdecode")
    @patch("hello.views.requests.get")
    def test_measure_includes_debug_on_measurement_error_when_requested(
        self, requests_get, imdecode, object_measurer
    ):
        measurer = self.configure_successful_measurement(
            requests_get, imdecode, object_measurer
        )
        measurer.debug = {
            "status": "failed",
            "errors": [{"code": "boom", "message": "measurement blew up"}],
        }
        measurer.measure.side_effect = RuntimeError("measurement blew up")

        with self.assertLogs("django.request", level="ERROR"):
            response = self.client.get(
                "/",
                {
                    "url": "https://example.com/photo.jpg",
                    "debug": "1",
                },
            )

        self.assertEqual(response.status_code, 500)
        payload = response.json()
        self.assertEqual(payload["error"], "Image measurement failed")
        self.assertEqual(payload["debug"]["status"], "failed")
        self.assertEqual(payload["debug"]["errors"][0]["code"], "boom")

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

    @patch("hello.views.ObjectMeasurer")
    @patch("hello.views.cv2.imdecode")
    @patch("hello.views.requests.get")
    def test_measure_accepts_scale(
        self, requests_get, imdecode, object_measurer
    ):
        self.configure_successful_measurement(requests_get, imdecode, object_measurer)

        response = self.client.get(
            "/",
            {
                "url": "https://example.com/photo.jpg",
                "reference_size_mm": "215.9,279.4",
                "scale": "2",
            },
        )

        self.assertEqual(response.status_code, 200)
        object_measurer.assert_called_once_with(
            reference_size_mm=(215.9, 279.4),
            scale=2,
        )

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
    def test_measure_rejects_invalid_measurer(self, requests_get):
        response = self.client.get(
            "/",
            {
                "url": "https://example.com/photo.jpg",
                "measurer": "mystery",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Invalid measurement option")
        self.assertIn("measurer", response.json()["details"])
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

    @patch("hello.views.requests.get")
    def test_measure_rejects_invalid_scale(self, requests_get):
        response = self.client.get(
            "/",
            {
                "url": "https://example.com/photo.jpg",
                "scale": "0",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Invalid measurement option")
        self.assertIn("greater than 0", response.json()["details"])
        requests_get.assert_not_called()
