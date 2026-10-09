import base64
import io
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, override_settings

from .views import (
    BadDxfUpload,
    DxfConversionDependencyMissing,
    DxfSvgOptions,
    PORTRAIT_POSTER_BOARD_MM,
    _load_ezdxf_drawing_modules,
    get_measurement_kwargs,
)


def dxf_svg_converter_available():
    try:
        _load_ezdxf_drawing_modules()
    except DxfConversionDependencyMissing:
        return False

    return True


class MeasureEndpointTest(SimpleTestCase):
    SVG_PATH_TOKEN_PATTERN = re.compile(
        r"[MmLlHhVvZz]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    )

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
    def test_measure_wraps_object_contour_path_as_svg_document(
        self, requests_get, imdecode, object_measurer
    ):
        measurer = self.configure_successful_measurement(
            requests_get, imdecode, object_measurer
        )
        measurer.debug = {
            "object_contour_svg": "M 10 20 L 30 20 L 30 40 Z",
            "imgWarp": np.zeros((60, 80, 3), dtype=np.uint8),
        }

        response = self.client.get("/", {"url": "https://example.com/photo.jpg"})

        self.assertEqual(response.status_code, 200)
        raw_path = "M 10 20 L 30 20 L 30 40 Z"
        svg = response.json()["svg"]
        self.assertNotEqual(svg, raw_path)
        self.assertTrue(svg.startswith("<svg "))
        self.assertTrue(svg.endswith("</svg>"))
        self.assertIn('viewBox="10 20 20 20"', svg)
        self.assertIn(f'<path d="{raw_path}"', svg)

    @patch("hello.views.ObjectMeasurer")
    @patch("hello.views.cv2.imdecode")
    @patch("hello.views.requests.get")
    def test_measure_returns_wrapped_contour_svg_in_debug_payload(
        self, requests_get, imdecode, object_measurer
    ):
        measurer = self.configure_successful_measurement(
            requests_get, imdecode, object_measurer
        )
        measurer.debug = {
            "object_contour_svg": "M 10 20 L 30 20 L 30 40 Z",
            "imgWarp": np.zeros((60, 80, 3), dtype=np.uint8),
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
        self.assertEqual(payload["debug"]["object_contour_svg"], payload["svg"])
        self.assertTrue(payload["debug"]["object_contour_svg"].startswith("<svg "))
        self.assertIn(
            'viewBox="10 20 20 20"',
            payload["debug"]["object_contour_svg"],
        )

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
                "scale": "0.001",
            },
        )

        self.assertEqual(response.status_code, 200)
        object_measurer.assert_called_once_with(
            reference_size_mm=(215.9, 279.4),
            scale=0.001,
        )

    def test_measurement_kwargs_accept_fractional_scale(self):
        kwargs = get_measurement_kwargs(
            {
                "reference_size_mm": "215.9,279.4",
                "scale": "0.001",
            }
        )

        self.assertEqual(
            kwargs,
            {
                "reference_size_mm": (215.9, 279.4),
                "scale": 0.001,
            },
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

    @patch("hello.views.convert_uploaded_dxf_to_svg")
    def test_dxf_to_svg_accepts_upload_with_default_scaling(self, convert_dxf):
        convert_dxf.return_value = '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        upload = SimpleUploadedFile(
            "part.dxf",
            b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n",
            content_type="application/dxf",
        )

        response = self.client.post("/dxf-to-svg/", {"file": upload})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["Content-Type"],
            "image/svg+xml; charset=utf-8",
        )
        self.assertEqual(
            response.headers["Content-Disposition"],
            'inline; filename="part.svg"',
        )
        self.assertEqual(response.headers["X-DXF-SVG-Scale"], "1.0")
        self.assertEqual(response.headers["X-DXF-SVG-Units"], "px")
        self.assertEqual(response.content.decode(), convert_dxf.return_value)

        uploaded_file, options = convert_dxf.call_args.args
        self.assertEqual(uploaded_file.name, "part.dxf")
        self.assertEqual(options, DxfSvgOptions())

    @patch("hello.views.convert_uploaded_dxf_to_svg")
    def test_dxf_to_svg_accepts_custom_scaling_options(self, convert_dxf):
        convert_dxf.return_value = '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        upload = SimpleUploadedFile(
            "drawing.dxf",
            b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n",
            content_type="application/dxf",
        )

        response = self.client.post(
            "/dxf-to-svg/?page_width=500",
            {
                "file": upload,
                "scale": "2.5",
                "fit_page": "yes",
                "margin": "4",
                "page_height": "600",
                "units": "mm",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            convert_dxf.call_args.args[1],
            DxfSvgOptions(
                scale=2.5,
                fit_page=True,
                margin=4.0,
                page_width=500.0,
                page_height=600.0,
                units="mm",
            ),
        )

    @patch("hello.views.convert_uploaded_dxf_to_svg")
    def test_dxf_to_svg_rejects_missing_upload(self, convert_dxf):
        response = self.client.post("/dxf-to-svg/", {"scale": "1"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Missing DXF upload")
        convert_dxf.assert_not_called()

    @patch("hello.views.convert_uploaded_dxf_to_svg")
    def test_dxf_to_svg_rejects_invalid_scale(self, convert_dxf):
        upload = SimpleUploadedFile(
            "part.dxf",
            b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n",
            content_type="application/dxf",
        )

        response = self.client.post("/dxf-to-svg/", {"file": upload, "scale": "0"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Invalid DXF conversion option")
        self.assertIn("greater than 0", response.json()["details"])
        convert_dxf.assert_not_called()

    @patch("hello.views.convert_uploaded_dxf_to_svg")
    def test_dxf_to_svg_reports_invalid_dxf_upload(self, convert_dxf):
        convert_dxf.side_effect = BadDxfUpload(
            "Uploaded file is not a readable DXF document."
        )
        upload = SimpleUploadedFile(
            "not-dxf.txt",
            b"not a dxf",
            content_type="text/plain",
        )

        response = self.client.post("/dxf-to-svg/", {"file": upload})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Invalid DXF upload")
        self.assertIn("not a readable DXF", response.json()["details"])

    @unittest.skipUnless(
        dxf_svg_converter_available(),
        "ezdxf drawing dependencies are not installed",
    )
    def test_dxf_to_svg_converts_real_dxf_when_ezdxf_is_available(self):
        import ezdxf

        doc = ezdxf.new()
        doc.modelspace().add_lwpolyline(
            [(0, 0), (10, 0), (10, 5), (0, 5), (0, 0)]
        )
        stream = io.StringIO()
        doc.write(stream)
        upload = SimpleUploadedFile(
            "rectangle.dxf",
            stream.getvalue().encode("utf-8"),
            content_type="application/dxf",
        )

        response = self.client.post("/dxf-to-svg/", {"file": upload})

        self.assertEqual(response.status_code, 200, response.content.decode())
        svg = response.content.decode()
        self.assertIn("<svg", svg)
        self.assertIn("</svg>", svg)

    @unittest.skipUnless(
        dxf_svg_converter_available(),
        "ezdxf drawing dependencies are not installed",
    )
    def test_dxf_to_svg_scales_path_geometry(self):
        dxf_bytes = self.build_rectangle_dxf_bytes(width=1000, height=500)
        path_sizes = {}

        for scale in (1, 0.1, 0.001):
            upload = SimpleUploadedFile(
                "rectangle.dxf",
                dxf_bytes,
                content_type="application/dxf",
            )

            response = self.client.post(
                "/dxf-to-svg/",
                {
                    "file": upload,
                    "scale": str(scale),
                },
            )

            self.assertEqual(response.status_code, 200, response.content.decode())
            svg_markup = response.content.decode()
            print(f"\n--- DXF to SVG scale={scale} ---\n{svg_markup}\n")
            path_sizes[scale] = self.svg_first_path_size(svg_markup)

        base_width, base_height = path_sizes[1]
        for scale in (0.1, 0.001):
            width, height = path_sizes[scale]
            self.assertAlmostEqual(width, base_width * scale)
            self.assertAlmostEqual(height, base_height * scale)

    @staticmethod
    def build_rectangle_dxf_bytes(width, height):
        import ezdxf

        doc = ezdxf.new()
        doc.modelspace().add_lwpolyline(
            [(0, 0), (width, 0), (width, height), (0, height), (0, 0)]
        )
        stream = io.StringIO()
        doc.write(stream)
        return stream.getvalue().encode("utf-8")

    @classmethod
    def svg_first_path_size(cls, svg_markup):
        root = ET.fromstring(svg_markup)

        for element in root.iter():
            if element.tag.endswith("path"):
                return cls.svg_path_size(element.attrib["d"])

        raise AssertionError("SVG did not contain a path")

    @classmethod
    def svg_path_size(cls, path_data):
        tokens = cls.SVG_PATH_TOKEN_PATTERN.findall(path_data)
        index = 0
        command = None
        x = 0.0
        y = 0.0
        start_x = 0.0
        start_y = 0.0
        points = []

        def is_command(token):
            return token.isalpha()

        def next_number():
            nonlocal index
            if index >= len(tokens) or is_command(tokens[index]):
                raise AssertionError(f"Unexpected SVG path data: {path_data!r}")

            value = float(tokens[index])
            index += 1
            return value

        while index < len(tokens):
            if is_command(tokens[index]):
                command = tokens[index]
                index += 1

            if command is None:
                raise AssertionError(f"Unexpected SVG path data: {path_data!r}")

            relative = command.islower()
            normalized = command.upper()

            if normalized == "Z":
                x = start_x
                y = start_y
                points.append((x, y))
                command = None
                continue

            while index < len(tokens) and not is_command(tokens[index]):
                if normalized in {"M", "L"}:
                    next_x = next_number()
                    next_y = next_number()
                    if relative:
                        next_x += x
                        next_y += y
                    x = next_x
                    y = next_y
                    points.append((x, y))

                    if normalized == "M":
                        start_x = x
                        start_y = y
                        command = "l" if relative else "L"
                        normalized = "L"
                elif normalized == "H":
                    next_x = next_number()
                    x = x + next_x if relative else next_x
                    points.append((x, y))
                elif normalized == "V":
                    next_y = next_number()
                    y = y + next_y if relative else next_y
                    points.append((x, y))
                else:
                    raise AssertionError(
                        f"Unsupported SVG path command in test: {command!r}"
                    )

        if not points:
            raise AssertionError("SVG path did not contain any points")

        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        return max(x_values) - min(x_values), max(y_values) - min(y_values)
