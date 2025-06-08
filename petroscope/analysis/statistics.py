from dataclasses import dataclass, asdict
import numpy as np
import json

from tqdm import tqdm

from petroscope.analysis.geometry import SegmPolygon, SegmPolygonData


@dataclass
class ClassStatistics:
    """Statistics for a single segmentation class.

    Areas are converted to real-world units using pixels_to_microns conversion.
    """

    id: int
    areas: list[float]  # Areas in real-world units (e.g., microns²)
    area_prc_of_image: float  # Percentage of total image area
    area_prc_of_classes: float  # Percentage of class area (no background)

    @property
    def count(self) -> int:
        return len(self.areas)

    @property
    def total_area(self) -> float:
        return sum(self.areas)

    @property
    def mean_area(self) -> float:
        return np.mean(self.areas)

    @property
    def std_area(self) -> float:
        return np.std(self.areas)


@dataclass
class IndividualObjectsStatistics:
    """Statistics for individual objects (not part of any group).

    Areas are converted to real-world units using pixels_to_microns conversion.
    """

    total_count: int  # Total number of individual objects
    areas: list[float]  # Areas in real-world units (e.g., microns²)
    total_area: float  # Sum of all individual object areas
    area_prc_of_image: float  # Percentage of total image area
    area_prc_of_classes: float  # Percentage of class area (no background)

    @property
    def mean_area(self) -> float:
        return np.mean(self.areas) if self.areas else 0.0

    @property
    def std_area(self) -> float:
        return np.std(self.areas) if self.areas else 0.0


@dataclass
class ClassesDistribution:
    """Distribution of classes by area and count."""

    class_area: dict[int, float]  # class_id -> total area
    class_count: dict[int, int]  # class_id -> total count
    class_area_prc: dict[int, float]  # class_id -> percentage of total area
    class_count_prc: dict[int, float]  # class_id -> percentage of total count
    area_bins: dict[str, dict[int, int]]  # size_category -> {class_id: count}

    @property
    def dominant_class_by_area(self) -> tuple[int, float]:
        """Returns (class_id, percentage) of the class with largest area."""
        return max(self.class_area_prc.items(), key=lambda x: x[1])

    @property
    def dominant_class_by_count(self) -> tuple[int, float]:
        """Returns (class_id, percentage) of the class with most objects."""
        return max(self.class_count_prc.items(), key=lambda x: x[1])


@dataclass
class SegmentationAnalysisResults:
    """Complete results of segmentation analysis."""

    classes_distribution: ClassesDistribution
    class_statistics: dict[int, ClassStatistics]
    individual_objects_statistics: dict[int, IndividualObjectsStatistics] = (
        None
    )
    classes: object = None  # Store classes object directly
    individual_objects: list[int] = None  # Objects not connected to others
    object_groups: list[set[int]] = None  # Groups with multiple objects

    def to_json(self, filepath: str) -> None:
        """Save analysis results to JSON file."""
        # Convert dataclasses to dictionaries
        classset = None
        if self.classes:
            classset = {
                "classes": [asdict(cls) for cls in self.classes.classes],
            }

        data = {
            "classes_distribution": asdict(self.classes_distribution),
            "class_statistics": {
                str(cls_id): asdict(stats)
                for cls_id, stats in self.class_statistics.items()
            },
            "classset": classset,
            "individual_objects_statistics": (
                {
                    str(cls_id): asdict(stats)
                    for cls_id, stats in (
                        self.individual_objects_statistics.items()
                    )
                }
                if self.individual_objects_statistics
                else None
            ),
            "individual_objects": self.individual_objects,
            "object_groups": (
                [list(group) for group in self.object_groups]
                if self.object_groups
                else None
            ),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> "SegmentationAnalysisResults":
        """Load analysis results from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        # Reconstruct ClassesDistribution
        classes_dist_data = data["classes_distribution"]
        classes_distribution = ClassesDistribution(
            class_area={
                int(k): v for k, v in classes_dist_data["class_area"].items()
            },
            class_count={
                int(k): v for k, v in classes_dist_data["class_count"].items()
            },
            class_area_prc={
                int(k): v
                for k, v in classes_dist_data["class_area_prc"].items()
            },
            class_count_prc={
                int(k): v
                for k, v in classes_dist_data["class_count_prc"].items()
            },
            area_bins={
                size_cat: {int(k): v for k, v in bins.items()}
                for size_cat, bins in classes_dist_data["area_bins"].items()
            },
        )

        # Reconstruct ClassStatistics
        class_statistics = {}
        for cls_id_str, stats_data in data["class_statistics"].items():
            cls_id = int(cls_id_str)
            class_statistics[cls_id] = ClassStatistics(
                id=stats_data["id"],
                areas=stats_data["areas"],
                area_prc_of_image=stats_data.get("area_prc_of_image", 0.0),
                area_prc_of_classes=stats_data.get("area_prc_of_classes", 0.0),
            )

        # Reconstruct ClassSet if available
        classset = None
        if data.get("classset"):
            from petroscope.segmentation.classes import ClassSet, Class

            classset_data = data["classset"]
            class_list = classset_data.get("classes", [])
            classset = ClassSet(
                [Class(**class_data) for class_data in class_list]
            )

        # Reconstruct IndividualObjectsStatistics if available
        individual_objects_statistics = None
        if data.get("individual_objects_statistics"):
            individual_objects_statistics = {}
            for cls_id_str, stats_data in data[
                "individual_objects_statistics"
            ].items():
                cls_id = int(cls_id_str)
                individual_objects_statistics[cls_id] = (
                    IndividualObjectsStatistics(
                        total_count=stats_data["total_count"],
                        areas=stats_data["areas"],
                        total_area=stats_data["total_area"],
                        area_prc_of_image=stats_data["area_prc_of_image"],
                        area_prc_of_classes=stats_data["area_prc_of_classes"],
                    )
                )

        # Reconstruct individual_objects and object_groups
        individual_objects = data.get("individual_objects")
        object_groups = None
        if data.get("object_groups"):
            object_groups = [set(group) for group in data["object_groups"]]

        return cls(
            classes_distribution=classes_distribution,
            class_statistics=class_statistics,
            classes=classset,
            individual_objects_statistics=individual_objects_statistics,
            individual_objects=individual_objects,
            object_groups=object_groups,
        )


class SegmentationStatisticsAnalyzer:
    def __init__(self, contact_distance_pixels: float = 2):
        self.contact_distance_pixels = contact_distance_pixels

    def analyze(self, data: SegmPolygonData) -> SegmentationAnalysisResults:
        """
        Analyze segmentation data and generate comprehensive statistics.

        Args:
            data: SegmPolygonData containing polygons and metadata

        Returns:
            SegmentationAnalysisResults with distribution and class statistics
        """
        if not data:
            raise ValueError("SegmPolygonData cannot be None")

        if not data.img_shape or len(data.img_shape) != 2:
            raise ValueError("Invalid image shape in SegmPolygonData")

        if data.img_shape[0] <= 0 or data.img_shape[1] <= 0:
            raise ValueError("Image dimensions must be positive")

        # Calculate class distribution
        cls_distribution = self._calculate_classes_distribution(data)

        # Calculate class statistics
        classes_statistics = self._calculate_class_statistics(data)

        # Extract connected groups of polygons
        groups = self._extract_groups(data.polygons)

        # Separate individual objects from groups using list comprehensions
        individual_objects = [
            polygon.id
            for group in groups
            for polygon in group
            if len(group) == 1
        ]
        object_groups = [
            {polygon.id for polygon in group}
            for group in groups
            if len(group) > 1
        ]

        # Calculate individual objects statistics
        calc_method = self._calculate_individual_objects_statistics
        individual_objects_statistics = calc_method(data, individual_objects)

        return SegmentationAnalysisResults(
            classes_distribution=cls_distribution,
            class_statistics=classes_statistics,
            classes=data.classes,
            individual_objects=individual_objects,
            object_groups=object_groups,
            individual_objects_statistics=individual_objects_statistics,
        )

    def _calculate_classes_distribution(
        self, data: SegmPolygonData
    ) -> ClassesDistribution:
        """
        Calculate distribution of classes by area and count.

        Returns:
            ClassesDistribution with percentage breakdowns
        """
        # Convert pixel area to real-world units
        unit_conversion_factor = data.pixels_to_microns**2
        total_image_area = (
            data.img_shape[0] * data.img_shape[1] * unit_conversion_factor
        )
        total_segmented_area = 0
        total_count = 0
        class_areas = {}
        class_counts = {}

        # Calculate totals and per-class values for segmented classes
        for cls_code, polygons in data.polygons_by_class.items():
            areas = [p.polygon.area * unit_conversion_factor for p in polygons]
            class_area = sum(areas)
            class_count = len(polygons)

            class_areas[cls_code] = class_area
            class_counts[cls_code] = class_count
            total_segmented_area += class_area
            total_count += class_count

        # Add background class (0)
        background_area = total_image_area - total_segmented_area
        class_areas[0] = background_area
        class_counts[0] = 1  # Consider background as one object

        # Calculate percentages based on total image area
        area_percentages = {
            cls_code: (area / total_image_area * 100)
            for cls_code, area in class_areas.items()
        }

        # Calculate count percentages
        total_count_with_bg = total_count + 1
        count_percentages = {}
        for cls_code, count in class_counts.items():
            if cls_code == 0:  # Background
                count_percentages[cls_code] = (
                    (1 / total_count_with_bg * 100)
                    if total_count_with_bg > 0
                    else 0
                )
            else:
                count_percentages[cls_code] = (
                    (count / total_count_with_bg * 100)
                    if total_count_with_bg > 0
                    else 0
                )

        # Calculate area bins (small, medium, large objects)
        area_bins = self._calculate_area_bins(data)

        return ClassesDistribution(
            class_area=class_areas,
            class_count=class_counts,
            class_area_prc=area_percentages,
            class_count_prc=count_percentages,
            area_bins=area_bins,
        )

    def _calculate_class_statistics(
        self, data: SegmPolygonData
    ) -> dict[int, ClassStatistics]:
        """
        Calculate area statistics for each class using ClassStatistics.

        Returns:
            Dictionary with ClassStatistics objects
        """
        class_stats = {}
        unit_conversion_factor = data.pixels_to_microns**2

        # Calculate total areas for percentage calculations
        total_image_area = (
            data.img_shape[0] * data.img_shape[1] * unit_conversion_factor
        )
        total_class_area = sum(
            sum(p.polygon.area * unit_conversion_factor for p in polygons)
            for polygons in data.polygons_by_class.values()
        )

        # Calculate statistics for segmented classes
        for cls_code, segm_polygons in data.polygons_by_class.items():
            areas = [
                p.polygon.area * unit_conversion_factor for p in segm_polygons
            ]
            class_total_area = sum(areas)

            # Calculate percentages
            area_prc_of_image = (
                (class_total_area / total_image_area * 100)
                if total_image_area > 0
                else 0.0
            )
            area_prc_of_classes = (
                (class_total_area / total_class_area * 100)
                if total_class_area > 0
                else 0.0
            )

            class_stats[cls_code] = ClassStatistics(
                id=cls_code,
                areas=areas,
                area_prc_of_image=area_prc_of_image,
                area_prc_of_classes=area_prc_of_classes,
            )

        # Add background class (0) statistics if not already present
        if 0 not in class_stats:
            total_segmented_area = sum(
                sum(p.polygon.area * unit_conversion_factor for p in polygons)
                for polygons in data.polygons_by_class.values()
            )
            background_area = total_image_area - total_segmented_area

            # Background percentages
            bg_area_prc_of_image = (
                (background_area / total_image_area * 100)
                if total_image_area > 0
                else 0.0
            )
            # Background is not part of class area, so percentage is 0
            bg_area_prc_of_classes = 0.0

            class_stats[0] = ClassStatistics(
                id=0,
                areas=[background_area],
                area_prc_of_image=bg_area_prc_of_image,
                area_prc_of_classes=bg_area_prc_of_classes,
            )

        return class_stats

    def _calculate_area_bins(
        self, data: SegmPolygonData
    ) -> dict[str, dict[int, int]]:
        """
        Categorize objects by area size within each class.

        Returns:
            Dictionary with size categories and class counts
        """
        # Apply unit conversion
        unit_conversion_factor = data.pixels_to_microns**2

        # Calculate global area percentiles for binning
        all_areas = [
            p.polygon.area * unit_conversion_factor for p in data.polygons
        ]
        if not all_areas:
            return {"small": {}, "medium": {}, "large": {}}

        area_33 = np.percentile(all_areas, 33)
        area_67 = np.percentile(all_areas, 67)

        bins = {"small": {}, "medium": {}, "large": {}}

        for cls_code, segm_polygons in data.polygons_by_class.items():
            small_count = medium_count = large_count = 0

            for p in segm_polygons:
                area = p.polygon.area * unit_conversion_factor
                if area <= area_33:
                    small_count += 1
                elif area <= area_67:
                    medium_count += 1
                else:
                    large_count += 1

            bins["small"][cls_code] = small_count
            bins["medium"][cls_code] = medium_count
            bins["large"][cls_code] = large_count

        return bins

    def _extract_groups(
        self, polygons: list[SegmPolygon]
    ) -> list[set[SegmPolygon]]:
        """Extract connected groups of polygons based on contact distance."""

        def _contact(sp: SegmPolygon, sp_other: SegmPolygon) -> bool:
            return (
                sp.polygon.distance(sp_other.polygon)
                < self.contact_distance_pixels
                or sp.polygon.intersects(sp_other.polygon)
                or sp.polygon.touches(sp_other.polygon)
            )

        # Find connected components using Union-Find
        polygon_groups = []
        visited = set()

        for sp in tqdm(polygons, "Finding connected groups"):
            if sp.id in visited:
                continue

            # Start a new group with this polygon
            current_group = {sp}
            stack = [sp]
            visited.add(sp.id)

            # Find all connected polygons using DFS
            while stack:
                current = stack.pop()
                for sp_other in polygons:
                    if sp_other.id in visited:
                        continue
                    if _contact(current, sp_other):
                        current_group.add(sp_other)
                        stack.append(sp_other)
                        visited.add(sp_other.id)

            polygon_groups.append(current_group)

        return polygon_groups

    def _calculate_individual_objects_statistics(
        self, data: SegmPolygonData, individual_objects: list[int]
    ) -> dict[int, IndividualObjectsStatistics]:
        """
        Calculate statistics for individual objects per class.

        Args:
            data: SegmPolygonData containing polygons and metadata
            individual_objects: List of polygon IDs that are individual objects

        Returns:
            Dictionary with IndividualObjectsStatistics per class
        """
        unit_conversion_factor = data.pixels_to_microns**2

        # Group individual objects by class
        individual_by_class = {}
        for polygon in data.polygons:
            if polygon.id in individual_objects:
                cls_id = polygon.cls_id
                if cls_id not in individual_by_class:
                    individual_by_class[cls_id] = []
                area = polygon.polygon.area * unit_conversion_factor
                individual_by_class[cls_id].append(area)

        # Calculate total areas for percentage calculations
        total_image_area = (
            data.img_shape[0] * data.img_shape[1] * unit_conversion_factor
        )
        total_class_area = sum(
            sum(p.polygon.area * unit_conversion_factor for p in polygons)
            for polygons in data.polygons_by_class.values()
        )

        # Calculate statistics for each class
        class_stats = {}
        for cls_id, areas in individual_by_class.items():
            total_individual_area = sum(areas)

            # Calculate percentages
            area_prc_of_image = (
                (total_individual_area / total_image_area * 100)
                if total_image_area > 0
                else 0.0
            )
            area_prc_of_classes = (
                (total_individual_area / total_class_area * 100)
                if total_class_area > 0
                else 0.0
            )

            class_stats[cls_id] = IndividualObjectsStatistics(
                total_count=len(areas),
                areas=areas,
                total_area=total_individual_area,
                area_prc_of_image=area_prc_of_image,
                area_prc_of_classes=area_prc_of_classes,
            )

        return class_stats
