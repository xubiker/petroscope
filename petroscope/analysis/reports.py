from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime

from petroscope.analysis.statistics import SegmentationAnalysisResults

# PDF imports - will be used conditionally
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Image,
        PageBreak,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors as rl_colors

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class AnalysisReportGenerator:
    """
    Generates comprehensive reports from segmentation analysis results.

    Creates beautiful high-resolution visualizations including:
    - Distribution pie charts (area and count)
    - Area bins visualization
    - Size distribution histograms
    - Individual objects analysis charts and histograms

    All visualizations are generated in high resolution (4K) by default.
    """

    def __init__(self):
        # Set plotly theme for beautiful plots
        pio.templates.default = "plotly_white"

    def generate_report(
        self,
        analysis_data: SegmentationAnalysisResults,
        output_dir: str = ".",
        generate_pdf: bool = False,
    ) -> None:
        """
        Generate complete analysis report with visualizations.

        Args:
            analysis_data: Results from segmentation analysis
            output_dir: Directory to save visualization files
            generate_pdf: Whether to create a PDF report

        Raises:
            ValueError: If no ClassSet with colors is provided
        """
        # Validate that ClassSet with colors is available
        if not analysis_data.classes or not analysis_data.classes.classes:
            raise ValueError(
                "Cannot generate report: No ClassSet provided. "
                "Analysis data must include class definitions with colors."
            )

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Generate and save visualizations with high resolution
        self.save_distribution_charts(analysis_data, output_path)
        self.save_size_histograms(analysis_data, output_path)

        # Generate individual objects visualizations if data exists
        if analysis_data.individual_objects_statistics:
            self.save_individual_objects_charts(analysis_data, output_path)

        # Generate connected object groups visualizations if data exists
        if analysis_data.connected_object_groups_statistics:
            self.save_connected_objects_charts(analysis_data, output_path)

        # Generate PDF report if requested
        if generate_pdf:
            self._generate_pdf_report(analysis_data, output_path)

        print(f"✅ Analysis report generated in: {output_path}")
        if generate_pdf:
            print(f"📄 PDF report: {output_path / 'analysis_report.pdf'}")

    def save_distribution_charts(
        self,
        analysis_data: SegmentationAnalysisResults,
        output_path: Path,
    ) -> None:
        """
        Create and save distribution charts (pie charts + area bins).

        Args:
            analysis_data: Analysis results
            output_path: Path to save the chart
        """
        fig = self._create_distribution_charts(analysis_data)

        # High resolution settings (4K)
        width, height, scale = 3200, 1600, 3

        # Save as PNG
        img_path = output_path / "distribution_charts.png"
        pio.write_image(fig, img_path, width=width, height=height, scale=scale)
        print(f"📊 Distribution charts saved: {img_path}")

    def save_size_histograms(
        self,
        analysis_data: SegmentationAnalysisResults,
        output_path: Path,
    ) -> None:
        """
        Create and save size distribution histograms for each class.

        Args:
            analysis_data: Analysis results
            output_path: Path to save the histograms
        """
        fig = self._create_size_histograms(analysis_data)

        # High resolution settings (4K)
        width, height, scale = 3200, 2000, 3

        # Save as PNG
        img_path = output_path / "size_histograms.png"
        pio.write_image(fig, img_path, width=width, height=height, scale=scale)
        print(f"📈 Size histograms saved: {img_path}")

    def save_individual_objects_charts(
        self,
        analysis_data: SegmentationAnalysisResults,
        output_path: Path,
    ) -> None:
        """
        Create and save individual objects analysis charts.

        Args:
            analysis_data: Analysis results with individual objects statistics
            output_path: Path to save the charts
        """
        # Create individual objects overview charts (pie chart + summary table)
        fig = self._create_individual_objects_charts(analysis_data)

        # High resolution settings (4K)
        width, height, scale = 3200, 1600, 3

        # Save as PNG
        img_path = output_path / "individual_objects_charts.png"
        pio.write_image(fig, img_path, width=width, height=height, scale=scale)
        print(f"📊 Individual objects charts saved: {img_path}")

        # Create and save individual objects histograms
        self.save_individual_objects_histograms(analysis_data, output_path)

    def save_individual_objects_histograms(
        self,
        analysis_data: SegmentationAnalysisResults,
        output_path: Path,
    ) -> None:
        """
        Create and save individual objects size distribution histograms.

        Args:
            analysis_data: Analysis results with individual objects statistics
            output_path: Path to save the histograms
        """
        fig = self._create_individual_objects_histograms(analysis_data)

        # High resolution settings (4K)
        width, height, scale = 3200, 2000, 3

        # Save as PNG
        hist_path = output_path / "individual_objects_histograms.png"
        pio.write_image(
            fig, hist_path, width=width, height=height, scale=scale
        )
        print(f"📊 Individual objects histograms saved: {hist_path}")

    def save_connected_objects_charts(
        self,
        analysis_data: SegmentationAnalysisResults,
        output_path: Path,
    ) -> None:
        """
        Create and save connected object groups analysis charts.

        Args:
            analysis_data: Analysis results with connected groups statistics
            output_path: Path to save the charts
        """
        # Create connected objects overview charts
        fig = self._create_connected_objects_charts(analysis_data)

        # High resolution settings (4K)
        width, height, scale = 3200, 1600, 3

        # Save as PNG
        img_path = output_path / "connected_objects_charts.png"
        pio.write_image(fig, img_path, width=width, height=height, scale=scale)
        print(f"🔗 Connected objects charts saved: {img_path}")

        # Create and save connected objects histograms
        self.save_connected_objects_histograms(analysis_data, output_path)

    def save_connected_objects_histograms(
        self,
        analysis_data: SegmentationAnalysisResults,
        output_path: Path,
    ) -> None:
        """
        Create and save connected objects size distribution histograms.

        Args:
            analysis_data: Analysis results with connected groups statistics
            output_path: Path to save the histograms
        """
        fig = self._create_connected_objects_histograms(analysis_data)

        # High resolution settings (4K)
        width, height, scale = 3200, 2000, 3

        # Save as PNG
        hist_path = output_path / "connected_objects_histograms.png"
        pio.write_image(
            fig, hist_path, width=width, height=height, scale=scale
        )
        print(f"📊 Connected objects histograms saved: {hist_path}")

    def _create_distribution_charts(
        self, analysis_data: SegmentationAnalysisResults
    ) -> go.Figure:
        """Create distribution charts with pie charts and area bins."""

        # Create subplots: 2 pie charts + 1 area bins chart
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=[
                "Area Distribution",
                "Count Distribution",
                "Area Bins",
                "",
                "",
                "",
            ],
            specs=[
                [{"type": "domain"}, {"type": "domain"}, {"type": "bar"}],
                [{"colspan": 3}, None, None],
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
        )

        # Get class information
        class_data = self._extract_class_data(analysis_data)

        # Add area pie chart
        self._add_area_pie_chart(fig, analysis_data, class_data, row=1, col=1)

        # Add count pie chart
        self._add_count_pie_chart(fig, analysis_data, class_data, row=1, col=2)

        # Add area bins chart
        self._add_area_bins_chart(fig, analysis_data, class_data, row=1, col=3)

        # Update layout
        fig.update_layout(
            title={
                "text": "Segmentation Analysis - Distribution Overview",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 48, "family": "Arial Black"},
            },
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5,
                font=dict(size=48),
            ),
            font=dict(size=48),
        )

        # Update subplot title fonts
        fig.update_annotations(font_size=48)

        return fig

    def _create_size_histograms(
        self, analysis_data: SegmentationAnalysisResults
    ) -> go.Figure:
        """Create size distribution histograms for each mineral class."""

        class_data = self._extract_class_data(analysis_data)

        # Filter out background class (code 0) for histograms
        mineral_classes = {}
        for code, data in class_data.items():
            if (
                code != 0
                and code in analysis_data.class_statistics
                and len(analysis_data.class_statistics[code].areas) > 0
            ):
                mineral_classes[code] = data

        if not mineral_classes:
            # Create empty figure if no mineral classes
            fig = go.Figure()
            fig.add_annotation(
                text="No mineral classes found",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=48),
            )
            return fig

        # Calculate subplot layout
        n_classes = len(mineral_classes)
        cols = min(3, n_classes)
        rows = (n_classes + cols - 1) // cols

        # Create subplot titles
        subplot_titles = [
            f"{data['label']}" for data in mineral_classes.values()
        ]

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
        )

        # Add histograms
        for i, (class_code, class_info) in enumerate(mineral_classes.items()):
            row = (i // cols) + 1
            col = (i % cols) + 1

            areas = analysis_data.class_statistics[class_code].areas

            # Calculate bins for better visualization
            n_bins = min(50, max(10, len(areas) // 5))

            fig.add_trace(
                go.Histogram(
                    x=areas,
                    name=class_info["label"],
                    marker_color=class_info["color_hex"],
                    opacity=0.8,
                    nbinsx=n_bins,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # Update axes for this subplot with larger fonts
            fig.update_xaxes(
                title_text="Area (μm²)" if row == rows else "",
                title_font=dict(size=48),
                tickfont=dict(size=48),
                row=row,
                col=col,
            )
            fig.update_yaxes(
                title_text="Count" if col == 1 else "",
                title_font=dict(size=48),
                tickfont=dict(size=48),
                row=row,
                col=col,
            )

        # Update layout
        fig.update_layout(
            title={
                "text": "Size Distribution by Mineral Class",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 48, "family": "Arial Black"},
            },
            height=400 * rows,
            font=dict(size=48),
        )

        # Update subplot title fonts
        fig.update_annotations(font_size=48)

        return fig

    def _create_individual_objects_histograms(
        self, analysis_data: SegmentationAnalysisResults
    ) -> go.Figure:
        """Create size distribution histograms for individual objects."""
        class_data = self._extract_class_data(analysis_data)

        # Filter classes that have individual objects data
        mineral_classes = {}
        for code, data in class_data.items():
            individual_stats = analysis_data.individual_objects_statistics
            if (
                code != 0  # Exclude background
                and code in individual_stats
                and len(individual_stats[code].areas) > 0
            ):
                mineral_classes[code] = data

        if not mineral_classes:
            # Create empty figure if no mineral classes
            fig = go.Figure()
            fig.add_annotation(
                text="No mineral classes found",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=48),
            )
            return fig

        # Calculate subplot layout
        n_classes = len(mineral_classes)
        cols = min(3, n_classes)
        rows = (n_classes + cols - 1) // cols

        # Create subplot titles
        subplot_titles = [
            f"{data['label']} (Individual Objects)"
            for data in mineral_classes.values()
        ]

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
        )

        # Add histograms for each class
        for i, (class_code, class_info) in enumerate(mineral_classes.items()):
            row = (i // cols) + 1
            col = (i % cols) + 1

            # Get individual objects areas for this class
            individual_stats = analysis_data.individual_objects_statistics
            areas = individual_stats[class_code].areas

            # Calculate bins for better visualization
            n_bins = min(50, max(10, len(areas) // 5))

            fig.add_trace(
                go.Histogram(
                    x=areas,
                    name=class_info["label"],
                    marker_color=class_info["color_hex"],
                    opacity=0.8,
                    nbinsx=n_bins,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # Update axes for this subplot with larger fonts
            fig.update_xaxes(
                title_text="Area (μm²)" if row == rows else "",
                title_font=dict(size=48),
                tickfont=dict(size=48),
                row=row,
                col=col,
            )
            fig.update_yaxes(
                title_text="Count" if col == 1 else "",
                title_font=dict(size=48),
                tickfont=dict(size=48),
                row=row,
                col=col,
            )

        # Update layout
        fig.update_layout(
            title={
                "text": "Individual Objects Size Distribution by Class",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 48, "family": "Arial Black"},
            },
            height=400 * rows,
            font=dict(size=48),
        )

        # Update subplot title fonts
        fig.update_annotations(font_size=48)

        return fig

    def _extract_class_data(
        self, analysis_data: SegmentationAnalysisResults
    ) -> dict:
        """
        Extract class information including colors and labels from ClassSet.

        Args:
            analysis_data: Results containing ClassSet with defined colors

        Returns:
            dict: Mapping of class codes to class information

        Note:
            Expects analysis_data.classes to be valid ClassSet with colors.
            Background class (code 0) is always added with default gray color.
        """
        class_data = {}

        # Add background class
        class_data[0] = {
            "label": "Background",
            "color_hex": "#2F2F2F",  # Dark gray
            "color_rgb": (47, 47, 47),
        }

        # Add mineral classes from ClassSet
        for cls in analysis_data.classes.classes:
            class_data[cls.code] = {
                "label": cls.label,
                "color_hex": cls.color,
                "color_rgb": cls.color_rgb,
            }

        return class_data

    def _add_area_pie_chart(self, fig, analysis_data, class_data, row, col):
        """Add area distribution pie chart."""
        dist = analysis_data.classes_distribution

        labels = []
        values = []
        colors = []

        for class_code, area in dist.class_area.items():
            if area > 0:  # Only include classes with area > 0
                labels.append(class_data[class_code]["label"])
                values.append(area)
                colors.append(class_data[class_code]["color_hex"])

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                name="Area Distribution",
                marker=dict(colors=colors, line=dict(color="white", width=2)),
                textinfo="label+percent",
                textposition="auto",
                textfont=dict(size=48),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Area: %{value:.2f} μm²<br>"
                    "Percentage: %{percent}<br>"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    def _add_count_pie_chart(self, fig, analysis_data, class_data, row, col):
        """Add count distribution pie chart."""
        dist = analysis_data.classes_distribution

        labels = []
        values = []
        colors = []

        for class_code, count in dist.class_count.items():
            if count > 0:  # Only include classes with count > 0
                labels.append(class_data[class_code]["label"])
                values.append(count)
                colors.append(class_data[class_code]["color_hex"])

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                name="Count Distribution",
                marker=dict(colors=colors, line=dict(color="white", width=2)),
                textinfo="label+percent",
                textposition="auto",
                textfont=dict(size=48),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Count: %{value}<br>"
                    "Percentage: %{percent}<br>"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    def _add_area_bins_chart(self, fig, analysis_data, class_data, row, col):
        """Add area bins (small/medium/large) stacked bar chart."""
        dist = analysis_data.classes_distribution

        # Get all class codes that have objects
        class_codes = set()
        for bin_name in ["small", "medium", "large"]:
            class_codes.update(dist.area_bins[bin_name].keys())

        # Filter out background for area bins
        class_codes = [code for code in class_codes if code != 0]

        if not class_codes:
            return

        # Prepare data for stacked bar chart
        small_counts = [
            dist.area_bins["small"].get(code, 0) for code in class_codes
        ]
        medium_counts = [
            dist.area_bins["medium"].get(code, 0) for code in class_codes
        ]
        large_counts = [
            dist.area_bins["large"].get(code, 0) for code in class_codes
        ]
        class_labels = [class_data[code]["label"] for code in class_codes]

        # Add traces for each size category
        fig.add_trace(
            go.Bar(
                x=class_labels,
                y=small_counts,
                name="Small",
                marker_color="#FFE5B4",
                legendgroup="bins",
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Bar(
                x=class_labels,
                y=medium_counts,
                name="Medium",
                marker_color="#FFB347",
                legendgroup="bins",
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Bar(
                x=class_labels,
                y=large_counts,
                name="Large",
                marker_color="#FF8C42",
                legendgroup="bins",
            ),
            row=row,
            col=col,
        )

        # Update axes
        fig.update_xaxes(
            title_text="Mineral Classes",
            title_font=dict(size=48),
            tickfont=dict(size=48),
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text="Object Count",
            title_font=dict(size=48),
            tickfont=dict(size=48),
            row=row,
            col=col,
        )

        # Set barmode to stack for this subplot
        fig.update_layout(barmode="stack")

    def _create_individual_objects_charts(
        self, analysis_data: SegmentationAnalysisResults
    ) -> go.Figure:
        """Create individual objects charts with pie and summary table."""
        if not analysis_data.individual_objects_statistics:
            # Create empty figure if no individual objects data
            fig = go.Figure()
            fig.add_annotation(
                text="No individual objects data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=48),
            )
            return fig

        # Create subplots: pie chart on top, summary table on bottom
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=[
                "Individual Objects Area Distribution",
                "Individual Objects Summary Statistics",
            ],
            specs=[
                [{"type": "domain"}],
                [{"type": "table"}],
            ],
            vertical_spacing=0.15,
        )

        # Get class information
        class_data = self._extract_class_data(analysis_data)

        # Add pie chart for individual objects area distribution
        self._add_individual_objects_pie_chart(
            fig, analysis_data, class_data, row=1, col=1
        )

        # Add summary table
        self._add_individual_objects_summary_table(
            fig, analysis_data, class_data, row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title={
                "text": "Individual Objects Analysis",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 48, "family": "Arial Black"},
            },
            height=1000,  # Increased height for vertical layout
            font=dict(size=48),
        )

        return fig

    def _add_individual_objects_pie_chart(
        self, fig, analysis_data, class_data, row, col
    ):
        """Add individual objects area distribution pie chart."""
        individual_stats = analysis_data.individual_objects_statistics

        labels = []
        values = []
        colors = []

        for class_code, stats in individual_stats.items():
            if stats.total_area > 0:  # Only include classes with area > 0
                labels.append(class_data[class_code]["label"])
                values.append(stats.total_area)
                colors.append(class_data[class_code]["color_hex"])

        if not values:
            # No data to display
            fig.add_annotation(
                text="No individual objects with area data",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.75,  # Position in the pie chart area (top half)
                showarrow=False,
                font=dict(size=48),
            )
            return

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                name="Individual Objects Area",
                marker=dict(colors=colors, line=dict(color="white", width=2)),
                textinfo="label+percent",
                textposition="auto",
                textfont=dict(size=48),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Area: %{value:.2f} μm²<br>"
                    "Percentage: %{percent}<br>"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    def _add_individual_objects_summary_table(
        self, fig, analysis_data, class_data, row, col
    ):
        """Add individual objects summary statistics table."""
        individual_stats = analysis_data.individual_objects_statistics

        if not individual_stats:
            return

        # Prepare table data
        headers = [
            "Class",
            "Count",
            "Total Area (μm²)",
            "Mean Area (μm²)",
            "Min Area (μm²)",
            "Max Area (μm²)",
            "Std Dev (μm²)",
        ]

        class_names = []
        counts = []
        total_areas = []
        mean_areas = []
        min_areas = []
        max_areas = []
        std_devs = []

        for class_code, stats in individual_stats.items():
            if stats.areas:  # Only include classes with data
                class_names.append(class_data[class_code]["label"])
                counts.append(stats.total_count)
                total_areas.append(f"{stats.total_area:.2f}")
                mean_areas.append(f"{stats.mean_area:.2f}")
                min_areas.append(f"{min(stats.areas):.2f}")
                max_areas.append(f"{max(stats.areas):.2f}")
                std_devs.append(f"{stats.std_area:.2f}")

        if not class_names:
            # No data to display
            fig.add_annotation(
                text="No individual objects statistics to display",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.25,  # Position in the table area (bottom half)
                showarrow=False,
                font=dict(size=48),
            )
            return

        # Create table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=headers,
                    fill_color="#4472C4",
                    font=dict(color="white", size=48),
                    align="center",
                    height=30,
                ),
                cells=dict(
                    values=[
                        class_names,
                        counts,
                        total_areas,
                        mean_areas,
                        min_areas,
                        max_areas,
                        std_devs,
                    ],
                    fill_color=[["#F2F2F2", "#FFFFFF"] * len(class_names)],
                    font=dict(size=48),
                    align="center",
                    height=50,
                ),
            ),
            row=row,
            col=col,
        )

    def _create_connected_objects_charts(
        self, analysis_data: SegmentationAnalysisResults
    ) -> go.Figure:
        """Create connected objects overview dashboard with charts."""
        if not analysis_data.connected_object_groups_statistics:
            # Create empty figure if no connected objects data
            fig = go.Figure()
            fig.add_annotation(
                text="No connected objects data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=48),
            )
            return fig

        # Create subplots: pie chart + bar chart + summary table
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Connected Groups by Total Area",
                "Connected Groups by Count",
                "Class Combination Frequency",
                "Summary Statistics",
            ],
            specs=[
                [{"type": "domain"}, {"type": "domain"}],
                [{"type": "bar"}, {"type": "table"}],
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        # Get class information
        class_data = self._extract_class_data(analysis_data)

        # Add pie chart for area distribution
        self._add_connected_objects_area_pie(
            fig, analysis_data, class_data, row=1, col=1
        )

        # Add pie chart for count distribution
        self._add_connected_objects_count_pie(
            fig, analysis_data, class_data, row=1, col=2
        )

        # Add frequency bar chart
        self._add_connected_objects_frequency_bar(
            fig, analysis_data, class_data, row=2, col=1
        )

        # Add summary table
        self._add_connected_objects_summary_table(
            fig, analysis_data, class_data, row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title={
                "text": "Connected Object Groups Analysis",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 48, "family": "Arial Black"},
            },
            height=1200,  # Increased height for 2x2 layout
            font=dict(size=48),
        )

        return fig

    def _create_connected_objects_histograms(
        self, analysis_data: SegmentationAnalysisResults
    ) -> go.Figure:
        """Create size distribution histograms for connected objects."""
        if not analysis_data.connected_object_groups_statistics:
            # Create empty figure if no connected objects data
            fig = go.Figure()
            fig.add_annotation(
                text="No connected objects data available for histograms",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=48),
            )
            return fig

        class_data = self._extract_class_data(analysis_data)
        connected_stats = analysis_data.connected_object_groups_statistics

        # Filter groups that have data
        valid_groups = {}
        for class_combo, stats in connected_stats.items():
            if len(stats.areas_per_connected_object) > 0:
                # Create a readable label for the class combination
                class_labels = []
                for cls_id in sorted(class_combo):
                    if cls_id in class_data:
                        class_labels.append(class_data[cls_id]["label"])
                combo_label = " + ".join(class_labels)
                valid_groups[combo_label] = stats

        if not valid_groups:
            # Create empty figure if no valid groups
            fig = go.Figure()
            fig.add_annotation(
                text="No connected object groups with area data",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=48),
            )
            return fig

        # Calculate subplot layout
        n_groups = len(valid_groups)
        cols = min(3, n_groups)
        rows = (n_groups + cols - 1) // cols

        # Create subplot titles
        subplot_titles = [f"{combo}" for combo in valid_groups.keys()]

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
        )

        # Add histograms for each group
        for i, (combo_label, stats) in enumerate(valid_groups.items()):
            row = (i // cols) + 1
            col = (i % cols) + 1

            areas = stats.areas_per_connected_object

            # Calculate bins for better visualization
            n_bins = min(50, max(10, len(areas) // 5))

            # Use a color from the first class in the combination
            first_cls_id = min(stats.class_combination)
            color = class_data.get(first_cls_id, {}).get(
                "color_hex", "#1f77b4"
            )

            fig.add_trace(
                go.Histogram(
                    x=areas,
                    name=combo_label,
                    marker_color=color,
                    opacity=0.8,
                    nbinsx=n_bins,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # Update axes for this subplot
            fig.update_xaxes(
                title_text="Area (μm²)" if row == rows else "",
                title_font=dict(size=42),
                tickfont=dict(size=42),
                row=row,
                col=col,
            )
            fig.update_yaxes(
                title_text="Count" if col == 1 else "",
                title_font=dict(size=42),
                tickfont=dict(size=42),
                row=row,
                col=col,
            )

        # Update layout
        fig.update_layout(
            title={
                "text": "Connected Objects Size Distribution by Combination",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 48, "family": "Arial Black"},
            },
            height=400 * rows,
            font=dict(size=42),
        )

        # Update subplot title fonts
        fig.update_annotations(font_size=42)

        return fig

    def _add_connected_objects_area_pie(
        self, fig, analysis_data, class_data, row, col
    ):
        """Add connected objects area distribution pie chart."""
        connected_stats = analysis_data.connected_object_groups_statistics

        labels = []
        values = []
        colors = []

        for class_combo, stats in connected_stats.items():
            if stats.total_area > 0:
                # Create readable label
                class_labels = []
                for cls_id in sorted(class_combo):
                    if cls_id in class_data:
                        class_labels.append(class_data[cls_id]["label"])
                combo_label = " + ".join(class_labels)

                labels.append(combo_label)
                values.append(stats.total_area)

                # Use color from the first class in combination
                first_cls_id = min(class_combo)
                color_hex = class_data.get(first_cls_id, {}).get(
                    "color_hex", "#1f77b4"
                )
                colors.append(color_hex)

        if not values:
            fig.add_annotation(
                text="No connected objects with area data",
                xref="paper",
                yref="paper",
                x=0.25,
                y=0.75,
                showarrow=False,
                font=dict(size=36),
            )
            return

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                name="Connected Area",
                marker=dict(colors=colors, line=dict(color="white", width=2)),
                textinfo="label+percent",
                textposition="auto",
                textfont=dict(size=36),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Area: %{value:.2f} μm²<br>"
                    "Percentage: %{percent}<br>"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    def _add_connected_objects_count_pie(
        self, fig, analysis_data, class_data, row, col
    ):
        """Add connected objects count distribution pie chart."""
        connected_stats = analysis_data.connected_object_groups_statistics

        labels = []
        values = []
        colors = []

        for class_combo, stats in connected_stats.items():
            if stats.connected_objects_count > 0:
                # Create readable label
                class_labels = []
                for cls_id in sorted(class_combo):
                    if cls_id in class_data:
                        class_labels.append(class_data[cls_id]["label"])
                combo_label = " + ".join(class_labels)

                labels.append(combo_label)
                values.append(stats.connected_objects_count)

                # Use color from the first class in combination
                first_cls_id = min(class_combo)
                color_hex = class_data.get(first_cls_id, {}).get(
                    "color_hex", "#1f77b4"
                )
                colors.append(color_hex)

        if not values:
            fig.add_annotation(
                text="No connected objects count data",
                xref="paper",
                yref="paper",
                x=0.75,
                y=0.75,
                showarrow=False,
                font=dict(size=36),
            )
            return

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                name="Connected Count",
                marker=dict(colors=colors, line=dict(color="white", width=2)),
                textinfo="label+percent",
                textposition="auto",
                textfont=dict(size=36),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Count: %{value}<br>"
                    "Percentage: %{percent}<br>"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    def _add_connected_objects_frequency_bar(
        self, fig, analysis_data, class_data, row, col
    ):
        """Add connected objects frequency bar chart."""
        connected_stats = analysis_data.connected_object_groups_statistics

        # Sort by frequency (count) for better visualization
        sorted_stats = sorted(
            connected_stats.items(),
            key=lambda x: x[1].connected_objects_count,
            reverse=True,
        )

        labels = []
        counts = []
        colors = []

        for class_combo, stats in sorted_stats:
            # Create readable label
            class_labels = []
            for cls_id in sorted(class_combo):
                if cls_id in class_data:
                    class_labels.append(class_data[cls_id]["label"])
            combo_label = " + ".join(class_labels)

            labels.append(combo_label)
            counts.append(stats.connected_objects_count)

            # Use color from the first class in combination
            first_cls_id = min(class_combo)
            color_hex = class_data.get(first_cls_id, {}).get(
                "color_hex", "#1f77b4"
            )
            colors.append(color_hex)

        if not labels:
            return

        fig.add_trace(
            go.Bar(
                x=labels,
                y=counts,
                name="Frequency",
                marker_color=colors,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Update axes
        fig.update_xaxes(
            title_text="Class Combinations",
            title_font=dict(size=36),
            tickfont=dict(size=32),
            tickangle=45,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text="Count",
            title_font=dict(size=36),
            tickfont=dict(size=36),
            row=row,
            col=col,
        )

    def _add_connected_objects_summary_table(
        self, fig, analysis_data, class_data, row, col
    ):
        """Add connected objects summary statistics table."""
        connected_stats = analysis_data.connected_object_groups_statistics

        if not connected_stats:
            return

        # Prepare table data
        headers = [
            "Class Combination",
            "Count",
            "Total Area (μm²)",
            "Mean Area (μm²)",
            "% of Image",
        ]

        combo_names = []
        counts = []
        total_areas = []
        mean_areas = []
        img_percentages = []

        for class_combo, stats in connected_stats.items():
            # Create readable label
            class_labels = []
            for cls_id in sorted(class_combo):
                if cls_id in class_data:
                    class_labels.append(class_data[cls_id]["label"])
            combo_label = " + ".join(class_labels)

            combo_names.append(combo_label)
            counts.append(stats.connected_objects_count)
            total_areas.append(f"{stats.total_area:.2f}")
            mean_areas.append(f"{stats.mean_connected_object_area:.2f}")
            img_percentages.append(f"{stats.area_prc_of_image:.2f}%")

        if not combo_names:
            fig.add_annotation(
                text="No connected objects statistics to display",
                xref="paper",
                yref="paper",
                x=0.75,
                y=0.25,
                showarrow=False,
                font=dict(size=36),
            )
            return

        # Create table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=headers,
                    fill_color="#4472C4",
                    font=dict(color="white", size=32),
                    align="center",
                    height=25,
                ),
                cells=dict(
                    values=[
                        combo_names,
                        counts,
                        total_areas,
                        mean_areas,
                        img_percentages,
                    ],
                    fill_color=[["#F2F2F2", "#FFFFFF"] * len(combo_names)],
                    font=dict(size=32),
                    align="center",
                    height=40,
                ),
            ),
            row=row,
            col=col,
        )

    def _generate_pdf_report(
        self, analysis_data: SegmentationAnalysisResults, output_path: Path
    ) -> None:
        """
        Generate a comprehensive PDF report with visualizations and statistics.

        Args:
            analysis_data: Results from segmentation analysis
            output_path: Path where images are saved and PDF will be created
        """
        if not PDF_AVAILABLE:
            print("❌ PDF generation requires reportlab package.")
            print("Install with: pip install reportlab")
            return

        pdf_path = output_path / "analysis_report.pdf"

        # Create PDF document
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch,
        )

        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=30,
            textColor=rl_colors.black,
            spaceAfter=30,
            alignment=1,  # Center alignment
        )

        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontSize=20,
            textColor=rl_colors.black,
            spaceAfter=12,
        )

        # Story elements
        story = []

        # Title page
        story.append(
            Paragraph("Mineral Segmentation Analysis Report", title_style)
        )
        story.append(Spacer(1, 20))
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(
            Paragraph(f"Generated on: {current_time}", styles["Normal"])
        )
        story.append(Spacer(1, 40))

        # Summary statistics
        story.append(Paragraph("Executive Summary", heading_style))

        dist = analysis_data.classes_distribution
        total_area = sum(dist.class_area.values())
        total_count = sum(dist.class_count.values())
        bg_area = dist.class_area.get(0, 0)
        bg_percent = bg_area / total_area * 100
        num_classes = len(analysis_data.class_statistics) - 1

        summary_text = f"""
        <b>Total Analysis Area:</b> {total_area:.2f} μm²<br/>
        <b>Total Objects Detected:</b> {total_count}<br/>
        <b>Number of Mineral Classes:</b> {num_classes}<br/>
        <b>Background Area:</b> {bg_area:.2f} μm²
        ({bg_percent:.1f}%)
        """

        story.append(Paragraph(summary_text, styles["Normal"]))
        story.append(Spacer(1, 20))

        # Class breakdown
        story.append(Paragraph("Mineral Class Breakdown", heading_style))

        for class_code, stats in analysis_data.class_statistics.items():
            if class_code == 0:  # Skip background for detailed breakdown
                continue

            class_name = "Unknown Class"
            if analysis_data.classes:
                for cls in analysis_data.classes.classes:
                    if cls.code == class_code:
                        class_name = cls.label
                        break

            if stats.areas:  # Only show classes that have objects
                min_area = min(stats.areas)
                max_area = max(stats.areas)
                class_text = f"""
                <b>{class_name} (Class {class_code}):</b><br/>
                • Objects: {len(stats.areas)}<br/>
                • Total Area: {stats.total_area:.2f} μm²<br/>
                • Average Area: {stats.mean_area:.2f} μm²<br/>
                • Area Range: {min_area:.2f} - {max_area:.2f} μm²<br/>
                """
                story.append(Paragraph(class_text, styles["Normal"]))
                story.append(Spacer(1, 10))

        story.append(PageBreak())

        # Add visualizations
        story.append(Paragraph("Distribution Analysis", heading_style))

        # Distribution charts
        dist_chart_path = output_path / "distribution_charts.png"
        if dist_chart_path.exists():
            story.append(
                Image(str(dist_chart_path), width=7 * inch, height=3.5 * inch)
            )
            story.append(Spacer(1, 20))

        # Size histograms
        story.append(Paragraph("Size Distribution Analysis", heading_style))

        hist_chart_path = output_path / "size_histograms.png"
        if hist_chart_path.exists():
            story.append(
                Image(str(hist_chart_path), width=7 * inch, height=5 * inch)
            )

        # Individual objects analysis
        if analysis_data.individual_objects_statistics:
            story.append(Spacer(1, 20))
            story.append(
                Paragraph("Individual Objects Analysis", heading_style)
            )

            # Individual objects summary
            individual_stats = analysis_data.individual_objects_statistics
            total_individual_objects = sum(
                stats.total_count for stats in individual_stats.values()
            )
            total_individual_area = sum(
                stats.total_area for stats in individual_stats.values()
            )

            individual_summary = f"""
            <b>Total Individual Objects:</b> {total_individual_objects}<br/>
            <b>Total Area:</b> {total_individual_area:.2f} μm²
            """
            story.append(Paragraph(individual_summary, styles["Normal"]))
            story.append(Spacer(1, 15))

            # Individual objects charts
            chart_path = output_path / "individual_objects_charts.png"
            if chart_path.exists():
                story.append(
                    Image(
                        str(chart_path),
                        width=7 * inch,
                        height=3.5 * inch,
                    )
                )
                story.append(Spacer(1, 20))

            # Individual objects histograms
            hist_path = output_path / "individual_objects_histograms.png"
            if hist_path.exists():
                story.append(
                    Paragraph(
                        "Individual Objects Size Distribution", heading_style
                    )
                )
                story.append(
                    Image(
                        str(hist_path),
                        width=7 * inch,
                        height=5 * inch,
                    )
                )

        # Connected objects analysis
        if analysis_data.connected_object_groups_statistics:
            story.append(Spacer(1, 20))
            story.append(
                Paragraph("Connected Objects Analysis", heading_style)
            )

            # Connected objects summary
            connected_stats = analysis_data.connected_object_groups_statistics
            total_connected_objects = sum(
                stats.connected_objects_count
                for stats in connected_stats.values()
            )
            total_connected_area = sum(
                stats.total_area for stats in connected_stats.values()
            )

            connected_summary = f"""
            <b>Total Connected Objects:</b> {total_connected_objects}<br/>
            <b>Total Area:</b> {total_connected_area:.2f} μm²
            """
            story.append(Paragraph(connected_summary, styles["Normal"]))
            story.append(Spacer(1, 15))

            # Connected objects charts
            connected_chart_path = output_path / "connected_objects_charts.png"
            if connected_chart_path.exists():
                story.append(
                    Image(
                        str(connected_chart_path),
                        width=7 * inch,
                        height=3.5 * inch,
                    )
                )
                story.append(Spacer(1, 20))

            # Connected objects histograms
            hist_filename = "connected_objects_histograms.png"
            connected_hist_path = output_path / hist_filename
            if connected_hist_path.exists():
                story.append(
                    Paragraph(
                        "Connected Objects Size Distribution", heading_style
                    )
                )
                story.append(
                    Image(
                        str(connected_hist_path),
                        width=7 * inch,
                        height=5 * inch,
                    )
                )

        # Build PDF
        doc.build(story)
        print(f"📄 PDF report generated: {pdf_path}")
