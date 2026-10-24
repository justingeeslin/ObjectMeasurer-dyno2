from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings
from django.urls import reverse


@override_settings(STORAGES={
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
})
class DemoTests(SimpleTestCase):
    @patch("hello.views.requests.get")
    def test_demo_pages_render_without_calling_apis(self, request_get):
        for route, heading, endpoint in (
            ("measure-demo", "Object Measurer", "measure"),
            ("dxf-demo", "DXF to SVG", "dxf-to-svg"),
        ):
            with self.subTest(route=route):
                response = self.client.get(reverse(route))
                self.assertContains(response, heading)
                self.assertContains(response, f'action="{reverse(endpoint)}"')
                self.assertContains(response, 'id="demo-form"')
                self.assertContains(response, 'demos/playground.js')
        request_get.assert_not_called()

    @patch("hello.views.ReferenceSurfaceMeasurer")
    @patch("hello.views.ObjectMeasurer")
    @patch("hello.views.cv2.imdecode", return_value=object())
    @patch("hello.views.requests.get")
    def test_engine_selection_with_current_measurement_contract(
        self, request_get, imdecode, object_engine, reference_engine
    ):
        request_get.return_value = SimpleNamespace(ok=True, content=b"image")
        for name, engine, other in (
            (None, reference_engine, object_engine),
            ("default", reference_engine, object_engine),
            ("object", object_engine, reference_engine),
            ("reference_surface", reference_engine, object_engine),
        ):
            with self.subTest(engine=name):
                object_engine.reset_mock()
                reference_engine.reset_mock()
                engine.return_value.measure.return_value = (
                    [SimpleNamespace(width_mm=100, height_mm=200)],
                    {"object_contour_svg": "M 0 0 L 100 0 L 100 200 Z"},
                )
                params = {
                    "url": "https://example.com/photo.jpg",
                    "reference_size_mm": "215.9,279.4",
                    "scale": "0.5",
                    "debug": "1",
                }
                if name is not None:
                    params["measurer"] = name
                response = self.client.get(reverse("measure"), params)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["measurer"], "object" if name == "object" else "reference_surface")
                self.assertEqual(response.json()["width_mm"], 100)
                self.assertIn("debug", response.json())
                engine.assert_called_once_with(reference_size_mm=(215.9, 279.4), scale=0.5)
                other.assert_not_called()
