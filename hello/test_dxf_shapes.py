import io
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase

from .tests import dxf_svg_converter_available
from .views import SVG_NUMBER_PATTERN, _parse_svg_points


SVG_NS = "{http://www.w3.org/2000/svg}"


class SeparateShapesOptionTests(SimpleTestCase):
    @patch("hello.views.convert_uploaded_dxf_to_svg", return_value=["<svg/>"])
    def test_json_response_and_boolean_option_precedence(self, convert):
        response = self.client.post(
            "/dxf-to-svg/?separate_shapes=false",
            {
                "dxf": SimpleUploadedFile("my pieces.dxf", b"drawing"),
                "separate_shapes": "yes",
                "scale": "25.4",
                "units": "in",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "application/json")
        self.assertEqual(response.json(), {"svgs": ["<svg/>"]})
        self.assertEqual(
            response.headers["Content-Disposition"],
            'inline; filename="my_pieces.json"',
        )
        self.assertEqual(response.headers["X-DXF-SVG-Scale"], "25.4")
        self.assertEqual(response.headers["X-DXF-SVG-Units"], "in")
        self.assertTrue(convert.call_args.args[1].separate_shapes)

    @patch("hello.views.convert_uploaded_dxf_to_svg")
    def test_invalid_boolean_is_rejected_before_conversion(self, convert):
        response = self.client.post(
            "/dxf-to-svg/?separate_shapes=maybe",
            {"file": SimpleUploadedFile("drawing.dxf", b"drawing")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("separate_shapes", response.json()["details"])
        convert.assert_not_called()

    @patch("hello.views.convert_uploaded_dxf_to_svg", return_value="<svg/>")
    def test_false_option_keeps_svg_response(self, convert):
        response = self.client.post(
            "/dxf-to-svg/?separate_shapes=off",
            {"file": SimpleUploadedFile("drawing.dxf", b"drawing")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"<svg/>")
        self.assertTrue(response.headers["Content-Type"].startswith("image/svg+xml"))
        self.assertFalse(convert.call_args.args[1].separate_shapes)


@unittest.skipUnless(
    dxf_svg_converter_available(), "ezdxf drawing dependencies are not installed"
)
class SeparateShapesConversionTests(SimpleTestCase):
    def setUp(self):
        import ezdxf

        self.doc = ezdxf.new()
        self.modelspace = self.doc.modelspace()

    def convert(self, **options):
        stream = io.StringIO()
        self.doc.write(stream)
        response = self.client.post(
            "/dxf-to-svg/?separate_shapes=true",
            {
                "file": SimpleUploadedFile(
                    "pieces.dxf", stream.getvalue().encode("utf-8")
                ),
                **options,
            },
        )
        self.assertEqual(response.status_code, 200, response.content.decode())
        return [ET.fromstring(markup) for markup in response.json()["svgs"]]

    @staticmethod
    def add_rectangle(space, width, height, origin=(0, 0), **attribs):
        x, y = origin
        space.add_lwpolyline(
            [(x, y), (x + width, y), (x + width, y + height), (x, y + height)],
            close=True,
            dxfattribs=attribs,
        )

    def assert_independent_svg(self, root):
        self.assertEqual(root.tag, f"{SVG_NS}svg")
        self.assertIsNone(root.find(f"{SVG_NS}rect"))
        group = root.find(f"{SVG_NS}g")
        self.assertGreaterEqual(len(group), 1)
        self.assertNotIn("data-dxf-layer", ET.tostring(root, encoding="unicode"))
        self.assertNotIn("data-dxf-pattern-piece", ET.tostring(root, encoding="unicode"))
        style = "".join(root.find(f"{SVG_NS}defs").itertext())
        for shape in group:
            self.assertIn(f'.{shape.get("class")} ', style)
        return group

    def displayed_polygon_bounds(self, root):
        group = self.assert_independent_svg(root)
        a, b, c, d, e, f = [
            float(match.group())
            for match in SVG_NUMBER_PATTERN.finditer(group.get("transform"))
        ]
        points = [
            (a * x + c * y + e, b * x + d * y + f)
            for x, y in _parse_svg_points(group[0].get("points"))
        ]
        xs, ys = zip(*points)
        return min(xs), min(ys), max(xs), max(ys)

    def test_connected_lines_and_polygons_have_independent_scaled_bounds(self):
        corners = [(10000, 20000), (10010, 20000), (10010, 20005), (10000, 20005)]
        for start, end in zip(corners, corners[1:] + corners[:1]):
            self.modelspace.add_line(start, end)
        for _ in range(2):
            self.add_rectangle(self.modelspace, 20, 8, origin=(-1000, -2000))

        roots = self.convert(scale="2")

        self.assertEqual(len(roots), 2)
        for root, (width, height) in zip(roots, [(20, 10), (40, 16)]):
            self.assertEqual(root.get("viewBox"), f"0 0 {width} {height}")
            self.assertEqual(root.get("width"), f"{width}mm")
            for actual, expected in zip(
                self.displayed_polygon_bounds(root), (0, 0, width, height)
            ):
                self.assertAlmostEqual(actual, expected)

    def test_curved_paths_keep_curves_styles_and_precise_bounds(self):
        self.modelspace.add_circle((1000, -2000), radius=10)
        self.modelspace.add_circle((-1000, 2000), radius=3)

        roots = self.convert(scale="0.5")

        self.assertEqual(len(roots), 2)
        for root, diameter in zip(roots, [10, 3]):
            group = self.assert_independent_svg(root)
            self.assertEqual(root.get("viewBox"), f"0 0 {diameter} {diameter}")
            self.assertEqual(group[0].tag, f"{SVG_NS}path")
            self.assertIn("c", group[0].get("d").lower())
            self.assertEqual(group[0].get("vector-effect"), "non-scaling-stroke")

    def test_seam_cleanup_runs_before_splitting_and_keeps_small_marks(self):
        self.add_rectangle(self.modelspace, 20, 10, layer="M")
        self.add_rectangle(self.modelspace, 16, 8, origin=(2, 1), layer="M")
        self.add_rectangle(self.modelspace, 1, 1, origin=(5, 5), layer="M")

        roots = self.convert()

        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0].get("viewBox"), "0 0 20 10")
        self.assertEqual(len(self.assert_independent_svg(roots[0])), 2)

    def test_pattern_block_keeps_marks_across_layers_and_outside_the_outline(self):
        block = self.doc.blocks.new("MARKED_PIECE")
        self.add_rectangle(block, 20, 10, layer="1")
        self.add_rectangle(block, 16, 8, origin=(2, 1), layer="14")
        block.add_line((2, 5), (18, 5), dxfattribs={"layer": "7"})
        block.add_line((10, 10), (10, 11), dxfattribs={"layer": "4"})
        self.modelspace.add_blockref("MARKED_PIECE", (100, 200))

        roots = self.convert()

        self.assertEqual(len(roots), 1)
        self.assertEqual(len(self.assert_independent_svg(roots[0])), 3)
        self.assertEqual(roots[0].get("viewBox"), "0 0 20 11")

    def test_loose_outlines_keep_enclosed_lines_and_attached_notches(self):
        for x in (0, 100):
            self.add_rectangle(self.modelspace, 20, 10, origin=(x, 0))
            self.modelspace.add_line((x + 2, 5), (x + 18, 5))
            self.modelspace.add_line((x + 10, 10), (x + 10, 11))

        roots = self.convert()

        self.assertEqual(len(roots), 2)
        for root in roots:
            self.assertEqual(len(self.assert_independent_svg(root)), 3)
            self.assertEqual(root.get("viewBox"), "0 0 20 11")

    def test_connected_arc_outline_keeps_its_internal_mark(self):
        self.modelspace.add_arc((100, 200), radius=10, start_angle=0, end_angle=180)
        self.modelspace.add_arc((100, 200), radius=10, start_angle=180, end_angle=360)
        self.modelspace.add_line((95, 200), (105, 200))

        roots = self.convert()

        self.assertEqual(len(roots), 1)
        self.assertEqual(len(self.assert_independent_svg(roots[0])), 3)
        self.assertEqual(roots[0].get("viewBox"), "0 0 20 20")

    def test_nested_grading_layers_keep_their_own_internal_marks(self):
        self.add_rectangle(self.modelspace, 20, 10, layer="M")
        self.add_rectangle(self.modelspace, 16, 8, origin=(2, 1), layer="S")
        for layer in ("M", "S"):
            self.modelspace.add_line((4, 5), (16, 5), dxfattribs={"layer": layer})

        roots = self.convert()

        self.assertEqual(len(roots), 2)
        for root in roots:
            self.assertEqual(len(self.assert_independent_svg(root)), 2)

    def test_small_piece_far_from_origin_preserves_its_geometry(self):
        self.add_rectangle(self.modelspace, 1.25, 0.5, origin=(1e9, 1e9))

        roots = self.convert(scale="2")

        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0].get("viewBox"), "0 0 2.5 1")
        for actual, expected in zip(self.displayed_polygon_bounds(roots[0]), (0, 0, 2.5, 1)):
            self.assertAlmostEqual(actual, expected)

    def test_square_fixtures_return_three_pieces_with_their_grainlines(self):
        fixtures = Path(__file__).resolve().parent.parent / "test-dxf"
        for name, scale in [("squares mm.dxf", "1"), ("squares inches.dxf", "25.4")]:
            with self.subTest(fixture=name):
                response = self.client.post("/dxf-to-svg/", {
                    "file": SimpleUploadedFile(name, (fixtures / name).read_bytes()),
                    "separate_shapes": "true", "scale": scale,
                })
                self.assertEqual(response.status_code, 200, response.content.decode())
                roots = [ET.fromstring(markup) for markup in response.json()["svgs"]]
                self.assertEqual(len(roots), 3)
                for root in roots:
                    self.assertEqual(len(self.assert_independent_svg(root)), 2)
                self.assertEqual(
                    sorted(float(root.get("viewBox").split()[2]) for root in roots),
                    [25.4, 50.8, 76.2],
                )

    def test_insert_instances_and_nested_grading_layers_remain_separate(self):
        block = self.doc.blocks.new("PIECE")
        self.add_rectangle(block, 20, 10)
        self.modelspace.add_blockref("PIECE", (0, 0))
        self.modelspace.add_blockref("PIECE", (100, 200))
        self.add_rectangle(self.modelspace, 16, 8, origin=(2, 1), layer="SIZE_S")

        roots = self.convert()

        self.assertEqual(len(roots), 3)
        self.assertEqual(
            [root.get("viewBox") for root in roots],
            ["0 0 20 10", "0 0 20 10", "0 0 16 8"],
        )

    def test_page_size_margins_and_fit_apply_to_each_shape(self):
        self.add_rectangle(self.modelspace, 10, 5, origin=(100, 200))
        self.add_rectangle(self.modelspace, 5, 10, origin=(1000, 2000))

        roots = self.convert(
            scale="2", units="cm", page_width="10", page_height="6",
            margin="0.5", fit_page="true",
        )

        self.assertEqual(len(roots), 2)
        for root, expected in zip(roots, [(5, 7.5, 95, 52.5), (37.5, 5, 62.5, 55)]):
            self.assertEqual(root.get("viewBox"), "0 0 100 60")
            for actual, target in zip(self.displayed_polygon_bounds(root), expected):
                self.assertAlmostEqual(actual, target)

    def test_compound_filled_path_is_a_complete_svg(self):
        hatch = self.modelspace.add_hatch()
        hatch.paths.add_polyline_path([(0, 0), (10, 0), (10, 5), (0, 5)], is_closed=True)
        hatch.paths.add_polyline_path([(2, 1), (4, 1), (4, 2), (2, 2)], is_closed=True)

        roots = self.convert()

        self.assertEqual(len(roots), 1)
        group = self.assert_independent_svg(roots[0])
        self.assertEqual(roots[0].get("viewBox"), "0 0 10 5")
        self.assertEqual(group[0].tag, f"{SVG_NS}path")
        self.assertEqual(group[0].get("d").count("M"), 2)

    def test_horizontal_and_vertical_lines_get_nonzero_viewports(self):
        self.modelspace.add_line((100, 200), (110, 200))
        self.modelspace.add_line((300, 400), (300, 410))

        roots = self.convert()

        self.assertEqual(len(roots), 2)
        for root in roots:
            self.assert_independent_svg(root)
            dimensions = [float(number) for number in root.get("viewBox").split()[2:]]
            self.assertTrue(all(number > 0 for number in dimensions))
            self.assertIn(10, dimensions)

    def test_empty_or_text_only_drawings_return_an_empty_array(self):
        self.assertEqual(self.convert(), [])
        self.modelspace.add_text("Label only")
        self.assertEqual(self.convert(), [])
