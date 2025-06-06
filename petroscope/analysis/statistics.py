from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


from petroscope.analysis.geometry import MaskPolygonProcessor, SegmPolygon
from petroscope.segmentation.classes import ClassSet


@dataclass
class SegmentationAnalysisResults:
    """Complete results of segmentation analysis."""

    class_stats: dict
    total_area: float
    connected_groups: list
    groups_by_size: dict
    mask: np.ndarray


class SegmentationStatisticsAnalyzer:
    def __init__(
        self,
        mask_source: np.ndarray,
        classes: ClassSet,
        pixels_to_microns: float = 1.0,
        min_area_threshold: float = 30,
    ):
        self.mask_source = mask_source
        self.classes = classes
        self.pixels_to_microns = pixels_to_microns
        self.min_area_threshold = min_area_threshold
        self._preprocess()

    def _preprocess(self):
        self.polygons_dict = MaskPolygonProcessor.mask_to_polygons(
            self.mask_source
        )
        self.polygons_dict = (
            MaskPolygonProcessor._remove_small_objects_and_holes(
                self.polygons_dict,
                min_area_threshold=self.min_area_threshold,
            )
        )

        count = 0
        self.polygons = []
        for cls_code, polygons in self.polygons_dict.items():
            cls = self.classes.get_class_by_code(cls_code)
            if cls is None:
                raise ValueError(
                    f"Class with ID {cls_code} not found in ClassSet."
                )
            for p in polygons:
                self.polygons.append(
                    SegmPolygon(
                        id=count,
                        cls=cls,
                        cls_id=cls_code,
                        polygon=p,
                    )
                )
                count += 1

    def _calculate_class_statistics(self) -> dict:
        """
        Calculate area statistics for each class.

        Returns:
            dict: Dictionary with class statistics
        """
        class_stats = {}
        total_area = 0

        # Calculate statistics for each class
        for cls_code, polygons in self.polygons_dict.items():
            cls = self.classes.get_class_by_code(cls_code)
            if cls is None:
                continue

            areas = [p.area for p in polygons]
            class_total_area = sum(areas)
            total_area += class_total_area

            class_stats[cls_code] = {
                "name": cls.name,
                "color": cls.color,
                "total_area": class_total_area,
                "areas": areas,
                "count": len(areas),
            }

            print(
                f"Class {cls.name} (ID {cls_code}): {class_total_area:.2f} pixels, {len(areas)} objects"
            )

        return class_stats, total_area

    def _calculate_arrangment_statistics(
        self, dist_treshold: float = 2
    ) -> dict:

        def _adhesion(sp: SegmPolygon, sp_other: SegmPolygon) -> bool:
            return (
                sp.polygon.distance(sp_other.polygon) < dist_treshold
                or sp.polygon.intersects(sp_other.polygon)
                or sp.polygon.touches(sp_other.polygon)
            )

        # Find connected components using Union-Find
        polygon_groups = []
        visited = set()

        for sp in tqdm(self.polygons, "Finding connected groups"):
            if sp.id in visited:
                continue

            # Start a new group with this polygon
            current_group = {sp}
            stack = [sp]
            visited.add(sp.id)

            # Find all connected polygons using DFS
            while stack:
                current = stack.pop()
                for sp_other in self.polygons:
                    if sp_other.id in visited:
                        continue
                    if _adhesion(current, sp_other):
                        current_group.add(sp_other)
                        stack.append(sp_other)
                        visited.add(sp_other.id)

            polygon_groups.append(current_group)

        # Separate individual grains from groups
        individual_grains = []
        connected_groups = []

        groups_by_size = {
            "individ": individual_grains,  # Individual grains
            "xs": [],  # 2 polygons in group
            "s": [],  # 3-5 polygon in group
            "m": [],  # 6-10 polygons in group
            "l": [],  # 11-50 polygons in group
            "xl": [],  # 51+ polygons in group
        }
        for gr in polygon_groups:
            if len(gr) == 1:
                groups_by_size["individ"].append(gr)
            if len(gr) == 2:
                groups_by_size["xs"].append(gr)
            elif len(gr) <= 5:
                groups_by_size["s"].append(gr)
            elif len(gr) <= 10:
                groups_by_size["m"].append(gr)
            elif len(gr) <= 50:
                groups_by_size["l"].append(gr)
            else:
                groups_by_size["xl"].append(gr)

        for size, groups in groups_by_size.items():
            print(f"Group size '{size}': {len(groups)} groups")

        return connected_groups, groups_by_size

    def _create_subplot_layout(self, n_classes: int) -> tuple:
        """
        Calculate subplot layout dimensions.

        Args:
            n_classes: Number of classes

        Returns:
            tuple: (rows, cols, subplot_titles, specs)
        """
        cols = 3  # Fixed 3 columns
        # Calculate rows: first row has pie chart + 1 histogram, remaining histograms in subsequent rows
        remaining_histograms = n_classes - 1
        additional_rows = (
            (remaining_histograms + cols - 1) // cols
            if remaining_histograms > 0
            else 0
        )
        rows = 1 + additional_rows

        subplot_titles = ["Class Distribution"] + [
            f"{stats['name']}" for stats in self.class_stats.values()
        ]

        # Create specs with larger pie chart (spans 2 columns)
        specs = []
        # First row: pie chart spans 2 columns, then one histogram
        first_row = [{"type": "domain", "colspan": 2}, None, {"type": "xy"}]
        specs.append(first_row)

        # Remaining rows for histograms
        for _ in range(rows - 1):
            specs.append([{"type": "xy"}] * cols)

        return rows, cols, subplot_titles, specs

    def _add_pie_chart(self, fig, class_stats: dict):
        """
        Add pie chart for class distribution including background.

        Args:
            fig: Plotly figure object
            class_stats: Dictionary with class statistics
        """
        class_names = [stats["name"] for stats in class_stats.values()]
        class_areas = [stats["total_area"] for stats in class_stats.values()]
        class_colors = [stats["color"] for stats in class_stats.values()]

        # Calculate background area
        total_image_area = (
            self.mask_source.shape[0] * self.mask_source.shape[1]
        )
        total_class_area = sum(class_areas)
        background_area = total_image_area - total_class_area

        # Add background to the lists
        class_names.append("Background")
        class_areas.append(background_area)
        class_colors.append("#000000")  # Black for background

        fig.add_trace(
            go.Pie(
                labels=class_names,
                values=class_areas,
                name="Class Distribution",
                marker=dict(colors=class_colors),
                textinfo="label+percent",
                textposition="auto",
            ),
            row=1,
            col=1,
        )

    def _add_histograms(self, fig, class_stats: dict, cols: int):
        """
        Add histograms for each class to show area distributions.

        Args:
            fig: Plotly figure object
            class_stats: Dictionary with class statistics
            cols: Number of columns in subplot layout
        """
        histogram_index = 0
        for i, (cls_code, stats) in enumerate(class_stats.items()):
            # First histogram goes to (1, 3), then continue normally
            if histogram_index == 0:
                row, col = 1, 3
            else:
                row = 2 + (histogram_index - 1) // cols
                col = 1 + (histogram_index - 1) % cols

            histogram_index += 1

            # Calculate appropriate number of bins
            n_bins = min(100, max(20, len(stats["areas"]) // 3))

            fig.add_trace(
                go.Histogram(
                    x=stats["areas"],
                    name=stats["name"],
                    marker_color=stats["color"],
                    opacity=0.7,
                    nbinsx=n_bins,
                ),
                row=row,
                col=col,
            )

            # Update axis labels for this subplot
            fig.update_xaxes(title_text="Area (pixels)", row=row, col=col)
            fig.update_yaxes(
                title_text="Count",
                type="log",
                row=row,
                col=col,
                dtick=1,  # Show only powers of 10
            )

    def area_stats(self) -> bytes:
        """
        Get area statistics for each class in the segmentation mask.
        Creates a combined plot with pie chart for class distribution and histograms for each class.

        Returns:
            bytes: Plot as PNG bytes
        """
        # Calculate statistics
        class_stats, total_area = self._calculate_class_statistics()

        # Handle empty case
        n_classes = len(class_stats)
        if n_classes == 0:
            fig = go.Figure()
            return np.array(pio.to_image(fig, format="png"))

        # Store class_stats for use in helper methods
        self.class_stats = class_stats

        # Create subplot layout
        rows, cols, subplot_titles, specs = self._create_subplot_layout(
            n_classes
        )

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            specs=specs,
        )

        # Add visualizations
        self._add_pie_chart(fig, class_stats)
        self._add_histograms(fig, class_stats, cols)

        # Update layout
        fig.update_layout(
            title_text=f"Segmentation Analysis - Total Area: {total_area:.2f} pixels",
            showlegend=False,
            height=400 * rows,
            width=500 * cols,
        )

        # Convert to bytes using plotly
        img_bytes = pio.to_image(fig, format="png", engine="kaleido")

        return img_bytes

    @property
    def mask(self):
        """
        Convert polygons back to a segmentation mask.
        """
        return MaskPolygonProcessor.polygons_to_mask(
            self.polygons, self.mask_source.shape
        )
