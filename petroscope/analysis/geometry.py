"""
Utilities for converting segmentation masks to polygons and performing
polygon-based operations for geometric analysis.
"""

from dataclasses import dataclass
import numpy as np
import cv2
from shapely import MultiPolygon
from shapely.geometry import Polygon
from tqdm import tqdm
import json
from typing import Any, Dict


from petroscope.segmentation.classes import Class, ClassSet


@dataclass(frozen=True)
class SegmPolygon:
    """
    Represents a polygon in segmentation analysis.
    Contains the polygon, its class, and an identifier.
    """

    id: int
    cls: Class
    cls_id: int
    polygon: Polygon

    def __hash__(self):
        return hash(self.id)


@dataclass
class SegmPolygonData:
    classes: ClassSet
    polygons: list[SegmPolygon]
    polygons_by_class: dict[int, list[SegmPolygon]]
    img_shape: tuple[int, int]
    pixels_to_microns: float

    def save_json(self, path: str):
        """
        Save SegmPolygonData to GeoJSON format.

        Args:
            path: File path to save the GeoJSON
        """
        geojson_data = {
            "type": "FeatureCollection",
            "metadata": {
                "img_shape": list(self.img_shape),
                "pixels_to_microns": self.pixels_to_microns,
                "classes": {
                    str(cls.code): {
                        "name": cls.name,
                        "color": cls.color,
                        "code": int(cls.code),
                        "label": cls.label,
                    }
                    for cls in self.classes.classes
                },
            },
            "features": [],
        }

        for polygon in self.polygons:
            feature = {
                "type": "Feature",
                "geometry": self._polygon_to_geojson(polygon.polygon),
                "properties": {
                    "id": int(polygon.id),
                    "class_id": int(polygon.cls_id),
                    "area": float(polygon.polygon.area),
                },
            }
            geojson_data["features"].append(feature)

        with open(path, "w") as f:
            json.dump(geojson_data, f, indent=2)

    @classmethod
    def load_json(
        cls,
        path: str,
        classes: ClassSet = None,
        parse_classes_from_json: bool = True,
    ) -> "SegmPolygonData":
        """
        Load SegmPolygonData from GeoJSON format.

        Args:
            path: File path to the GeoJSON file
            classes: ClassSet containing class definitions (required
            if parse_classes_from_json is False)
            parse_classes_from_json: If True, parse classes from JSON
            metadata instead of using provided classes

        Returns:
            SegmPolygonData instance
        """
        with open(path, "r") as f:
            geojson_data = json.load(f)

        metadata = geojson_data.get("metadata", {})
        img_shape = tuple(metadata.get("img_shape", [0, 0]))
        pixels_to_microns = metadata.get("pixels_to_microns", 1.0)

        # Parse classes from JSON if requested
        if parse_classes_from_json:
            from petroscope.segmentation.classes import ClassSet, Class

            classes_data = metadata.get("classes", {})
            class_list = []
            for code_str, class_info in classes_data.items():
                class_obj = Class(
                    name=class_info["name"],
                    color=class_info["color"],
                    code=class_info["code"],
                    label=class_info["label"],
                )
                class_list.append(class_obj)
            classes = ClassSet(class_list)
        elif classes is None:
            raise ValueError(
                "Either provide 'classes' parameter "
                "or set 'parse_classes_from_json=True'"
            )

        polygons = []
        polygons_by_class = {}

        for feature in geojson_data.get("features", []):
            properties = feature.get("properties", {})
            geometry = feature.get("geometry", {})

            polygon_id = properties.get("id")
            cls_id = properties.get("class_id")
            class_label = properties.get("class_label")  # Parse class label

            # Get class from ClassSet
            class_obj = classes.get_class_by_code(cls_id)
            if class_obj is None:
                continue

            # Verify class label matches if available
            if class_label and class_obj.name != class_label:
                print(
                    f"Warning: Class label mismatch for ID {cls_id}. "
                    f"Expected '{class_obj.name}', found '{class_label}'"
                )

            # Convert GeoJSON geometry to Shapely polygon
            shapely_polygon = cls._geojson_to_polygon(geometry)
            if shapely_polygon is None:
                continue

            segm_polygon = SegmPolygon(
                id=polygon_id,
                cls=class_obj,
                cls_id=cls_id,
                polygon=shapely_polygon,
            )

            polygons.append(segm_polygon)

            if cls_id not in polygons_by_class:
                polygons_by_class[cls_id] = []
            polygons_by_class[cls_id].append(segm_polygon)

        return cls(
            classes=classes,
            polygons=polygons,
            polygons_by_class=polygons_by_class,
            img_shape=img_shape,
            pixels_to_microns=pixels_to_microns,
        )

    def _polygon_to_geojson(self, polygon: Polygon) -> Dict[str, Any]:
        """
        Convert Shapely Polygon to GeoJSON geometry.

        Args:
            polygon: Shapely Polygon

        Returns:
            GeoJSON geometry dictionary
        """
        # Get exterior coordinates and ensure they're [x, y] format and closed
        exterior_coords = list(polygon.exterior.coords)
        if exterior_coords[0] != exterior_coords[-1]:
            exterior_coords.append(exterior_coords[0])  # Close the ring

        coordinates = [exterior_coords]

        # Process holes (interiors)
        for interior in polygon.interiors:
            hole_coords = list(interior.coords)
            if hole_coords[0] != hole_coords[-1]:
                hole_coords.append(hole_coords[0])  # Close the ring
            coordinates.append(hole_coords)

        return {"type": "Polygon", "coordinates": coordinates}

    @staticmethod
    def _geojson_to_polygon(geometry: Dict[str, Any]) -> Polygon:
        """
        Convert GeoJSON geometry to Shapely Polygon.

        Args:
            geometry: GeoJSON geometry dictionary

        Returns:
            Shapely Polygon or None if conversion fails
        """
        if geometry.get("type") != "Polygon":
            return None

        coordinates = geometry.get("coordinates", [])
        if not coordinates:
            return None

        try:
            exterior = coordinates[0]
            holes = coordinates[1:] if len(coordinates) > 1 else []
            return Polygon(exterior, holes=holes)
        except Exception:
            return None

    def to_pixel_mask(self, colorize: bool = True) -> np.ndarray:
        """
        Convert polygons to a pixel mask.

        Args:
            colorize: If True, return a colored mask using class colors

        Returns:
            Segmentation mask as numpy array (grayscale if colorize=False,
            RGB if colorize=True)
        """

        def _fill_polygon_in_mask_with_value(
            mask: np.ndarray, polygon: Polygon, value: int
        ):
            """Fill a polygon area in a mask with a specific value."""
            # Convert polygon to contour format for OpenCV
            if polygon.exterior is None:
                return

            exterior_coords = np.array(polygon.exterior.coords, dtype=np.int32)

            # Fill the exterior
            cv2.fillPoly(mask, [exterior_coords], int(value))

            # Remove holes
            for hole in polygon.interiors:
                hole_coords = np.array(hole.coords, dtype=np.int32)
                cv2.fillPoly(
                    mask, [hole_coords], 0
                )  # Fill holes with background

        mask = np.zeros(self.img_shape, dtype=np.uint8)

        for segm_polygon in tqdm(self.polygons, "Converting polygons to mask"):
            polygon_mask = np.zeros(self.img_shape, dtype=np.uint8)
            _fill_polygon_in_mask_with_value(
                polygon_mask, segm_polygon.polygon, segm_polygon.cls_id
            )
            mask = np.maximum(mask, polygon_mask)

        if colorize:
            from petroscope.segmentation.vis import SegmVisualizer

            return SegmVisualizer.colorize_mask(
                mask, self.classes.code_to_color_rgb
            )

        return mask


class MaskPolygonProcessor:
    def __init__(self, classes: ClassSet, pixels_to_microns: float = 1.0):
        self.classes = classes
        self.pixels_to_microns = pixels_to_microns

    def extract_polygon_data(
        self,
        mask: np.ndarray,
        apply_morphology_closing: bool = False,
        min_area_threshold_pixels: float = 30,
        simplify_tolerance: float = 0.5,
    ) -> SegmPolygonData:
        polygons = self._extract_polygons(
            mask,
            apply_morphology_closing=apply_morphology_closing,
            simplify_tolerance=simplify_tolerance,
        )
        polygons = self._remove_small_objects_and_holes(
            polygons,
            min_area_threshold=min_area_threshold_pixels,
        )
        segm_polygons = []
        segm_polygons_by_class = {}
        count = 0
        for cls_id, polys in polygons.items():
            cls = self.classes.get_class_by_code(cls_id)
            segm_polygons_by_class[cls_id] = []
            if cls is None:
                raise ValueError(
                    f"Class with ID {cls_id} not found in ClassSet."
                )
            for p in polys:
                sp = SegmPolygon(
                    id=count,
                    cls=cls,
                    cls_id=cls_id,
                    polygon=p,
                )
                segm_polygons.append(sp)
                segm_polygons_by_class[cls_id].append(sp)
                count += 1

        return SegmPolygonData(
            classes=self.classes,
            polygons=segm_polygons,
            polygons_by_class=segm_polygons_by_class,
            img_shape=mask.shape,
            pixels_to_microns=self.pixels_to_microns,
        )

    def _extract_polygons(
        self,
        mask: np.ndarray,
        apply_morphology_closing: bool = False,
        simplify_tolerance: float = 0.5,
    ) -> dict[int, list[SegmPolygon]]:
        """
        Convert a segmentation mask polygons for each class.

        Args:
            mask: 2D numpy array with class labels

        Returns:
            Dictionary mapping class labels to lists of Polygon objects
        """
        if mask.ndim != 2:
            raise ValueError("Mask must be a 2D array")

        polygons_by_class = {}
        unique_classes = np.unique(mask)

        # Skip background (0) class
        unique_classes = unique_classes[(unique_classes != 0)]

        for class_id in unique_classes:
            # Create binary mask for this class
            binary_mask = (mask == class_id).astype(np.uint8)

            if apply_morphology_closing:
                # Apply morphological closing to fill small holes
                binary_mask = cv2.morphologyEx(
                    binary_mask,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                )

            # Find contours with hierarchy to detect holes
            contours, hierarchy = cv2.findContours(
                binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
            )

            if hierarchy is None or len(hierarchy) == 0:
                # No contours found, skip this class
                continue

            class_polygons = []

            # Process each contour
            for i, contour in enumerate(contours):

                coords = contour.squeeze()
                if len(coords) < 3:
                    # Not enough points to form a polygon
                    continue

                # Create polygon from exterior coordinates
                try:
                    exterior_coords = [
                        (point[0], point[1]) for point in coords
                    ]

                    # Check if this is an exterior contour
                    if (
                        hierarchy[0][i][3] == -1
                    ):  # No parent, so it's an exterior
                        holes = []

                        # Find holes for this contour
                        for j, other_contour in enumerate(contours):
                            if hierarchy[0][j][3] == i:
                                hole_coords = other_contour.squeeze()
                                if len(hole_coords) > 3:
                                    hole_coords_list = [
                                        (point[0], point[1])
                                        for point in hole_coords
                                    ]
                                    holes.append(hole_coords_list)

                    # Create polygon with holes
                    polygon = Polygon(exterior_coords, holes=holes)

                    if polygon.is_valid:
                        class_polygons.append(polygon)
                    else:

                        fixed = polygon.buffer(0)
                        if isinstance(fixed, Polygon):
                            if fixed.area != 0:
                                class_polygons.append(fixed)
                        elif isinstance(fixed, MultiPolygon):
                            cleaned_polygons = list(fixed.geoms)
                            class_polygons.extend(cleaned_polygons)
                        else:
                            print(
                                "Unexpected error while fixing polygon: "
                                f"{fixed} (class {class_id} contour {i})"
                            )
                            continue
                except Exception as e:
                    # Skip invalid polygons
                    print(f"Error creating polygon: {e}")
                    continue

            class_polygons = [
                p.simplify(simplify_tolerance, preserve_topology=True)
                for p in class_polygons
            ]

            if class_polygons:
                polygons_by_class[class_id] = class_polygons

        return polygons_by_class

    @staticmethod
    def _remove_small_objects_and_holes(
        polygons_by_class: dict[int, list[Polygon]],
        min_area_threshold: float,
    ) -> dict[int, list[Polygon]]:
        """
        Remove small objects by merging with intersecting/touching objects.

        Args:
            polygons_by_class: Dictionary mapping class labels to polygons
            min_area_threshold: Minimum area threshold for objects and holes

        Returns:
            Dictionary mapping class labels to lists of Polygon objects
        """
        result = {}
        for class_id, polygons in polygons_by_class.items():
            # leave only polygons with area >= min_area_threshold
            large_polygons = [
                p for p in polygons if p.area >= min_area_threshold
            ]
            # remove small holes
            result_polygons = []
            for p in large_polygons:
                interiors = [
                    i
                    for i in p.interiors
                    if Polygon(i).area >= min_area_threshold
                ]
                polygon = Polygon(p.exterior.coords, holes=interiors)
                result_polygons.append(polygon)

            if result_polygons:
                result[class_id] = result_polygons
        return result
