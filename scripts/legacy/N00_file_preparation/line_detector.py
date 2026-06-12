import cv2


class LineDetector:
    def __init__(self, settings):
        """Initialize the object and store configuration.
        
        Args:
            settings: Optional configuration dictionary used to override defaults.
        
        Returns:
            None.
        """
        self.settings = settings

    def create_black_mask(self, image):
        """Creates a mask of only very dark/black pixels.
        This is useful for detecting printed/form/table lines while ignoring blue handwriting.
        
        Args:
            image: Input image array.
        
        Returns:
            Binary dark-pixel mask.
        """

        # If the image is already grayscale, fallback to dark threshold.
        if len(image.shape) == 2:
            gray = image

            _, black_mask = cv2.threshold(
                gray,
                self.settings.get("gray_black_threshold", 80),
                255,
                cv2.THRESH_BINARY_INV
            )

            return black_mask

        # OpenCV loads color images as BGR.
        b, g, r = cv2.split(image)

        black_pixels = (
            (b < self.settings.get("black_blue_max", 90)) &
            (g < self.settings.get("black_green_max", 90)) &
            (r < self.settings.get("black_red_max", 90))
        )

        black_mask = black_pixels.astype("uint8") * 255

        return black_mask

    def create_threshold_mask(self, image):
        """Old-style behavior: invert a binary/grayscale image.
        This is kept as fallback.
        
        Args:
            image: Input image array.
        
        Returns:
            Binary threshold mask.
        """

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return cv2.bitwise_not(image)

    def create_line_source_mask(self, image):
        """Select the source mask used for line detection.
        
        Args:
            image: Input image array.
        
        Returns:
            Binary mask used for line detection.
        """
        mode = self.settings.get("line_detection_mode", "black_color")

        if mode == "black_color":
            return self.create_black_mask(image)

        if mode == "threshold":
            return self.create_threshold_mask(image)

        raise ValueError("line_detection_mode must be 'black_color' or 'threshold'")

    def detect_lines(self, image, kernel_size, min_length, direction):
        """Detect line bounding boxes for one direction.
        
        Args:
            image: Input image array.
            kernel_size: Morphology kernel size as (width, height).
            min_length: Minimum line length to keep.
            direction: Line direction or neighbor direction to evaluate.
        
        Returns:
            Tuple of line boxes and a binary line mask.
        """
        source_mask = self.create_line_source_mask(image)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            kernel_size
        )

        lines_mask = cv2.morphologyEx(
            source_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        contours, _ = cv2.findContours(
            lines_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        lines = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w <= 0 or h <= 0:
                continue

            if direction == "horizontal":
                aspect_ratio = w / max(h, 1)

                max_thickness = self.settings.get("max_horizontal_line_thickness", 12)
                min_aspect_ratio = self.settings.get("min_horizontal_aspect_ratio", 12)

                if w < min_length:
                    continue

                if h > max_thickness:
                    continue

                if aspect_ratio < min_aspect_ratio:
                    continue

            elif direction == "vertical":
                aspect_ratio = h / max(w, 1)

                max_thickness = self.settings.get("max_vertical_line_thickness", 12)
                min_aspect_ratio = self.settings.get("min_vertical_aspect_ratio", 8)

                if h < min_length:
                    continue

                if w > max_thickness:
                    continue

                if aspect_ratio < min_aspect_ratio:
                    continue

            else:
                raise ValueError("direction must be 'horizontal' or 'vertical'")

            lines.append((x, y, w, h))

        if direction == "horizontal":
            lines = sorted(lines, key=lambda line: line[1])
        else:
            lines = sorted(lines, key=lambda line: line[0])

        return lines, lines_mask

    def detect_horizontal_lines(self, image):
        """Detect horizontal structural lines.
        
        Args:
            image: Input image array.
        
        Returns:
            Tuple of horizontal line boxes and mask.
        """
        return self.detect_lines(
            image,
            self.settings.get("horizontal_kernel_size", (80, 1)),
            self.settings.get("min_horizontal_line_width", 300),
            "horizontal"
        )

    def detect_vertical_lines(self, image):
        """Detect vertical structural lines.
        
        Args:
            image: Input image array.
        
        Returns:
            Tuple of vertical line boxes and mask.
        """
        return self.detect_lines(
            image,
            self.settings.get("vertical_kernel_size", (1, 40)),
            self.settings.get("min_vertical_line_height", 40),
            "vertical"
        )

    def extract_horizontal_y_levels(self, lines, y_tolerance=8):
        """Merge horizontal line boxes into row y-levels.
        
        Args:
            lines: List of line bounding boxes.
            y_tolerance: Maximum y-distance for merging nearby levels.
        
        Returns:
            List of merged y-coordinate levels.
        """
        y_levels = []

        for x, y, w, h in lines:
            center_y = y + h // 2
            matched = False

            for index, existing_y in enumerate(y_levels):
                if abs(center_y - existing_y) <= y_tolerance:
                    y_levels[index] = (existing_y + center_y) // 2
                    matched = True
                    break

            if not matched:
                y_levels.append(center_y)

        return sorted(y_levels)

    def draw_horizontal_lines_preview(self, image, lines):
        """Draw horizontal line boxes on a preview image.
        
        Args:
            image: Input image array.
            lines: List of line bounding boxes.
        
        Returns:
            Preview image with horizontal lines drawn.
        """
        if len(image.shape) == 2:
            preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            preview = image.copy()

        for index, (x, y, w, h) in enumerate(lines, start=1):
            cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                preview,
                str(index),
                (x, max(y - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

        return preview

    def draw_vertical_lines_preview(self, image, lines):
        """Draw vertical line boxes on a preview image.
        
        Args:
            image: Input image array.
            lines: List of line bounding boxes.
        
        Returns:
            Preview image with vertical lines drawn.
        """
        if len(image.shape) == 2:
            preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            preview = image.copy()

        for index, (x, y, w, h) in enumerate(lines, start=1):
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(
                preview,
                str(index),
                (x + 2, max(y - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

        return preview

    def draw_combined_lines_preview(self, image, horizontal_lines, vertical_lines):
        """Draw both horizontal and vertical line boxes.
        
        Args:
            image: Input image array.
            horizontal_lines: Detected horizontal line boxes.
            vertical_lines: Detected vertical line boxes.
        
        Returns:
            Preview image with all lines drawn.
        """
        if len(image.shape) == 2:
            preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            preview = image.copy()

        for x, y, w, h in horizontal_lines:
            cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 0, 0), 2)

        for x, y, w, h in vertical_lines:
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 2)

        return preview