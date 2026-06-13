import cv2
import numpy as np


class ImagePreprocessor:
    def __init__(self, settings):
        """Initialize the object and store configuration.
        
        Args:
            settings: Optional configuration dictionary used to override defaults.
        
        Returns:
            None.
        """
        self.settings = settings

    def load_image(self, image_path):
        """Load an image from disk.
        
        Args:
            image_path: Path to the image file.
        
        Returns:
            Loaded image array.
        """
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not load image from path: {image_path}")

        return image

    def rotate_major(self, image):
        """Rotate an image by the configured major angle.
        
        Args:
            image: Input image array.
        
        Returns:
            Rotated image array.
        """
        angle = self.settings.get("manual_major_rotation", 0)

        if angle == 0:
            return image

        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        if angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)

        if angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        raise ValueError("manual_major_rotation must be 0, 90, 180, or 270")

    def rotate_by_angle(self, image, angle):
        """Rotate an image by an arbitrary deskew angle.

        Args:
            image: Input image array.
            angle: Rotation angle in degrees.

        Returns:
            Rotated image with the same size as the input.
        """
        if abs(angle) < self.settings.get("deskew_min_abs_angle", 0.2):
            return image

        if abs(angle) > self.settings.get("deskew_max_abs_angle", 10):
            return image

        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        return cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    def convert_to_grayscale(self, image):
        """Convert an image to grayscale if needed.
        
        Args:
            image: Input image array.
        
        Returns:
            Grayscale image array.
        """
        if len(image.shape) == 2:
            return image

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def denoise_image(self, gray):
        """Reduce noise in a grayscale image.
        
        Args:
            gray: Grayscale image array.
        
        Returns:
            Denoised grayscale image array.
        """
        return cv2.fastNlMeansDenoising(
            gray,
            None,
            h=self.settings.get("denoise_strength", 10),
            templateWindowSize=7,
            searchWindowSize=21
        )

    def threshold_image(self, denoised):
        """Create a binary thresholded image.
        
        Args:
            denoised: Denoised grayscale image.
        
        Returns:
            Binary thresholded image array.
        """
        return cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.settings.get("threshold_block_size", 35),
            self.settings.get("threshold_c", 14)
        )

    def deskew_image(self, thresholded):
        """Deskew using long horizontal line candidates.
        This is more reliable for forms than using all black pixels,
        because all black pixels include handwriting, printed text, stamps, and noise.
        
        Args:
            thresholded: Value used by this function.
        
        Returns:
            Tuple of deskewed image and detected skew angle.
        """

        inverted = cv2.bitwise_not(thresholded)

        height, width = thresholded.shape[:2]

        # Emphasize horizontal structures before Hough line detection.
        horizontal_kernel_width = self.settings.get(
            "deskew_horizontal_kernel_width",
            max(width // 4, 80)
        )

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (horizontal_kernel_width, 1)
        )

        horizontal_mask = cv2.morphologyEx(
            inverted,
            cv2.MORPH_OPEN,
            horizontal_kernel
        )

        def collect_angles(mask, hough_threshold, min_line_length, max_line_gap):
            """Collect near-horizontal Hough line angles from a mask.
            
            Args:
                mask: Binary mask to analyze.
                hough_threshold: Vote threshold for Hough line detection.
                min_line_length: Minimum Hough line length.
                max_line_gap: Maximum gap allowed inside a Hough line.
            
            Returns:
                List of detected near-horizontal angles in degrees.
            """
            lines = cv2.HoughLinesP(
                mask,
                rho=1,
                theta=np.pi / 180,
                threshold=hough_threshold,
                minLineLength=min_line_length,
                maxLineGap=max_line_gap
            )

            local_angles = []
            if lines is None:
                return local_angles

            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1

                if dx == 0:
                    continue

                angle = np.degrees(np.arctan2(dy, dx))

                # Keep only almost-horizontal lines.
                if abs(angle) <= self.settings.get("deskew_max_abs_angle", 10):
                    local_angles.append(angle)

            return local_angles

        angles = collect_angles(
            horizontal_mask,
            self.settings.get("deskew_hough_threshold", 80),
            self.settings.get("deskew_min_line_length", int(width * 0.35)),
            self.settings.get("deskew_max_line_gap", 25)
        )

        # If strict pass found no useful angles, retry with relaxed params.
        if len(angles) == 0:
            relaxed_kernel_width = self.settings.get(
                "deskew_relaxed_horizontal_kernel_width",
                max(width // 12, 60)
            )
            relaxed_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (relaxed_kernel_width, 1)
            )
            relaxed_mask = cv2.morphologyEx(
                inverted,
                cv2.MORPH_OPEN,
                relaxed_kernel
            )

            angles = collect_angles(
                relaxed_mask,
                self.settings.get("deskew_relaxed_hough_threshold", 35),
                self.settings.get("deskew_relaxed_min_line_length", int(width * 0.10)),
                self.settings.get("deskew_relaxed_max_line_gap", 35)
            )

        # Fallback: if no reliable horizontal lines found, keep old behavior.
        if len(angles) == 0:
            coords = cv2.findNonZero(inverted)

            if coords is None:
                return thresholded, 0

            angle = cv2.minAreaRect(coords)[-1]

            if angle < -45:
                angle = 90 + angle

            if abs(angle) > self.settings.get("deskew_max_abs_angle", 10):
                return thresholded, angle

            deskew_angle = angle
        else:
            # Median is safer than mean because handwriting/stamps can create outliers.
            deskew_angle = float(np.median(angles))

        # If the detected angle is tiny, don't rotate.
        if abs(deskew_angle) < self.settings.get("deskew_min_abs_angle", 0.2):
            return thresholded, deskew_angle

        # If angle is too large, likely bad detection.
        if abs(deskew_angle) > self.settings.get("deskew_max_abs_angle", 10):
            return thresholded, deskew_angle

        center = (width // 2, height // 2)

        # In this pipeline's coordinate convention, detected deskew_angle
        # already points to the correction direction.
        correction_angle = deskew_angle

        rotation_matrix = cv2.getRotationMatrix2D(
            center,
            correction_angle,
            1.0
        )

        deskewed = cv2.warpAffine(
            thresholded,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return deskewed, deskew_angle

    def get_white_margin_bounds(self, image):
        """Calculate the crop box that removes outer white margins.

        Args:
            image: Input image array.

        Returns:
            Tuple of x1, y1, x2, y2 crop coordinates.
        """
        padding = self.settings.get("crop_padding", 20)

        inverted = cv2.bitwise_not(image)
        coords = cv2.findNonZero(inverted)

        height, width = image.shape[:2]

        if coords is None:
            return 0, 0, width, height

        x, y, w, h = cv2.boundingRect(coords)

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, width)
        y2 = min(y + h + padding, height)

        return x1, y1, x2, y2

    def crop_to_bounds(self, image, bounds):
        """Crop an image to explicit x1, y1, x2, y2 bounds.

        Args:
            image: Input image array.
            bounds: Tuple of x1, y1, x2, y2 crop coordinates.

        Returns:
            Cropped image array.
        """
        x1, y1, x2, y2 = bounds
        return image[y1:y2, x1:x2]

    def crop_white_margins(self, image):
        """Remove white margins around foreground content.
        
        Args:
            image: Input image array.
        
        Returns:
            Cropped image array.
        """
        bounds = self.get_white_margin_bounds(image)
        return self.crop_to_bounds(image, bounds)
