import cv2


class RegionDetector:
    def __init__(self, settings):
        """Initialize the object and store configuration.
        
        Args:
            settings: Optional configuration dictionary used to override defaults.
        
        Returns:
            None.
        """
        self.settings = settings

    def detect_regions(self, image):
        """Detect broad merged content regions.
        
        Args:
            image: Input image array.
        
        Returns:
            Tuple of region boxes and region mask.
        """
        region_kernel_size = self.settings.get("region_kernel_size", (35, 9))
        region_min_area = self.settings.get("region_min_area", 5000)
        region_min_width = self.settings.get("region_min_width", 100)
        region_min_height = self.settings.get("region_min_height", 30)

        inverted = cv2.bitwise_not(image)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            region_kernel_size
        )

        merged = cv2.morphologyEx(
            inverted,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            merged,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area < region_min_area:
                continue

            if w < region_min_width or h < region_min_height:
                continue

            regions.append((x, y, w, h))

        regions = sorted(regions, key=lambda region: (region[1], region[0]))

        return regions, merged

    def draw_regions_preview(self, image, regions):
        """Draw detected regions on a preview image.
        
        Args:
            image: Input image array.
            regions: List of detected region boxes.
        
        Returns:
            Preview image with region boxes drawn.
        """
        preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for index, (x, y, w, h) in enumerate(regions, start=1):
            cv2.rectangle(
                preview,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                preview,
                str(index),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        return preview

    def run(self, image):
        """Run the detector/orchestrator entry point.
        
        Args:
            image: Input image array.
        
        Returns:
            Detector result dictionary or raises when unsupported.
        """
        regions, regions_mask = self.detect_regions(image)
        regions_preview = self.draw_regions_preview(image, regions)

        return {
            "regions": regions,
            "regions_mask": regions_mask,
            "regions_preview": regions_preview,
            "count": len(regions)
        }
