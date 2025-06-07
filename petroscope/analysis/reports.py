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

    Creates beautiful visualizations including:
    - Distribution pie charts (area and count)
    - Area bins visualization
    - Size distribution histograms
    """

    def __init__(self):
        # Set plotly theme for beautiful plots
        pio.templates.default = "plotly_white"

    def generate_report(
        self,
        analysis_data: SegmentationAnalysisResults,
        output_dir: str = ".",
        generate_pdf: bool = False,
        high_resolution: bool = True,
    ) -> None:
        """
        Generate complete analysis report with visualizations.

        Args:
            analysis_data: Results from segmentation analysis
            output_dir: Directory to save visualization files
            generate_pdf: Whether to create a PDF report
            high_resolution: Whether to use high resolution (4K) output

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

        # Generate and save visualizations with enhanced resolution
        self.save_distribution_charts(
            analysis_data, output_path, high_resolution=high_resolution
        )
        self.save_size_histograms(
            analysis_data, output_path, high_resolution=high_resolution
        )

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
        high_resolution: bool = True,
    ) -> None:
        """
        Create and save distribution charts (pie charts + area bins).

        Args:
            analysis_data: Analysis results
            output_path: Path to save the chart
            high_resolution: Whether to use high resolution output
        """
        fig = self._create_distribution_charts(analysis_data)

        # Enhanced resolution settings
        if high_resolution:
            width, height, scale = 3200, 1600, 3  # 4K resolution
        else:
            width, height, scale = 1600, 800, 2  # Standard resolution

        # Save as PNG
        img_path = output_path / "distribution_charts.png"
        pio.write_image(fig, img_path, width=width, height=height, scale=scale)
        print(f"📊 Distribution charts saved: {img_path}")

    def save_size_histograms(
        self,
        analysis_data: SegmentationAnalysisResults,
        output_path: Path,
        high_resolution: bool = True,
    ) -> None:
        """
        Create and save size distribution histograms for each class.

        Args:
            analysis_data: Analysis results
            output_path: Path to save the histograms
            high_resolution: Whether to use high resolution output
        """
        fig = self._create_size_histograms(analysis_data)

        # Enhanced resolution settings
        if high_resolution:
            width, height, scale = 3200, 2000, 3  # 4K resolution
        else:
            width, height, scale = 1600, 1000, 2  # Standard resolution

        # Save as PNG
        img_path = output_path / "size_histograms.png"
        pio.write_image(fig, img_path, width=width, height=height, scale=scale)
        print(f"📈 Size histograms saved: {img_path}")

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
                "font": {"size": 36, "family": "Arial Black"},
            },
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5,
                font=dict(size=20),
            ),
            font=dict(size=20),
        )

        # Update subplot title fonts
        fig.update_annotations(font_size=22)

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
                font=dict(size=28),
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
                title_font=dict(size=20),
                tickfont=dict(size=18),
                row=row,
                col=col,
            )
            fig.update_yaxes(
                title_text="Count" if col == 1 else "",
                title_font=dict(size=20),
                tickfont=dict(size=18),
                row=row,
                col=col,
            )

        # Update layout
        fig.update_layout(
            title={
                "text": "Size Distribution by Mineral Class",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 36, "family": "Arial Black"},
            },
            height=400 * rows,
            font=dict(size=20),
        )

        # Update subplot title fonts
        fig.update_annotations(font_size=22)

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
                textfont=dict(size=18),
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
                textfont=dict(size=18),
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
            title_font=dict(size=20),
            tickfont=dict(size=18),
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text="Object Count",
            title_font=dict(size=20),
            tickfont=dict(size=18),
            row=row,
            col=col,
        )

        # Set barmode to stack for this subplot
        fig.update_layout(barmode="stack")

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
            fontSize=24,
            textColor=rl_colors.black,
            spaceAfter=30,
            alignment=1,  # Center alignment
        )

        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontSize=16,
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

        # Build PDF
        doc.build(story)
        print(f"📄 PDF report generated: {pdf_path}")
