from pathlib import Path

import cv2
import numpy as np
from petroscope.analysis.reports import AnalysisReportGenerator
from petroscope.analysis.statistics import SegmentationAnalysisResults
from petroscope.segmentation.classes import ClassSet
from petroscope.analysis.geometry import (
    MaskPolygonProcessor,
    SegmPolygonData,
)
from petroscope.segmentation.vis import SegmVisualizer


def process_segmentation_mask(
    mask_path: Path,
    classes: ClassSet,
    output_dir: str = "./data/",
) -> None:

    # Load mask
    mask_src = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_src.ndim == 3:
        mask_src = mask_src[:, :, 0]

    # Convert mask using class mappings
    mask_tmp = np.zeros_like(mask_src, dtype=np.uint8)
    for v in np.unique(mask_src):
        if v > 0:
            mask_tmp[mask_src == v] = classes.idx_to_code[v]
    mask_src = mask_tmp

    mpp = MaskPolygonProcessor(classes=classes, pixels_to_microns=0.85)

    polygon_data = mpp.extract_polygon_data(
        mask=mask_src,
        min_area_threshold_pixels=30,
        simplify_tolerance=0.5,
    )
    polygon_data.save_json(
        str(Path(output_dir) / "polygon_data.json"),
    )

    polygon_data = SegmPolygonData.load_json(
        str(Path(output_dir) / "polygon_data.json"),
    )
    mask_new_colored = polygon_data.to_pixel_mask(colorize=True)

    # Create visualizations
    mask_src_colored = SegmVisualizer.colorize_mask(
        mask_src, classes.code_to_color_rgb
    )

    diff = np.abs(
        mask_src_colored.astype(np.float32)
        - mask_new_colored.astype(np.float32)
    ).astype(np.uint8)

    comparison = SegmVisualizer.compose(
        [mask_src_colored, mask_new_colored, diff],
        header_data="Original | Processed (polygon cleanup), Difference",
    )

    cv2.imwrite(str(Path(output_dir) / "mask_out.png"), comparison)


def perform_analysis(
    segm_data_path: Path,
    output_dir: str = "./data/",
) -> None:

    polygon_data = SegmPolygonData.load_json(
        str(Path(output_dir) / "polygon_data.json"),
    )

    from petroscope.analysis.statistics import SegmentationStatisticsAnalyzer

    analyzer = SegmentationStatisticsAnalyzer()
    results = analyzer.analyze(polygon_data)

    results.to_json(str(Path(output_dir) / "analysis_results.json"))

    results2 = SegmentationAnalysisResults.from_json(
        str(Path(output_dir) / "analysis_results.json")
    )

    reporter = AnalysisReportGenerator()
    reporter.generate_report(
        results2,
        output_dir=Path(output_dir) / "reports",
        generate_pdf=True,
    )


# Example usage
if __name__ == "__main__":
    from petroscope.segmentation.classes import LumenStoneClasses

    mask_path = Path("./data/019_x05.png")
    classes = LumenStoneClasses.S1_S2()

    process_segmentation_mask(
        mask_path=mask_path,
        classes=classes,
    )

    perform_analysis(
        segm_data_path=Path("./data/polygon_data.json"),
    )
