import cv2


class ImageQualityAnalyzer:
    def __init__(self, settings=None):
        """Initialize the object and store configuration.
        
        Args:
            settings: Optional configuration dictionary used to override defaults.
        
        Returns:
            None.
        """
        if settings is None:
            settings = {}

        self.settings = settings

    def calculate_blur_score(self, gray):
        """Calculate a Laplacian blur score.
        
        Args:
            gray: Grayscale image array.
        
        Returns:
            Numeric blur score.
        """
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return blur_score

    def calculate_brightness(self, gray):
        """Calculate average image brightness.
        
        Args:
            gray: Grayscale image array.
        
        Returns:
            Average brightness value.
        """
        brightness = gray.mean()
        return brightness

    def calculate_contrast(self, gray):
        """Calculate image contrast.
        
        Args:
            gray: Grayscale image array.
        
        Returns:
            Contrast value.
        """
        contrast = gray.std()
        return contrast

    def analyze(self, gray):
        """Calculate raw quality metrics for an image.
        
        Args:
            gray: Grayscale image array.
        
        Returns:
            Dictionary of raw quality metrics.
        """
        blur_score = self.calculate_blur_score(gray)
        brightness = self.calculate_brightness(gray)
        contrast = self.calculate_contrast(gray)

        return {
            "blur_score": blur_score,
            "brightness": brightness,
            "contrast": contrast
        }

    def analyze_rounded(self, gray):
        """Calculate rounded quality metrics for reporting.
        
        Args:
            gray: Grayscale image array.
        
        Returns:
            Dictionary of rounded quality metrics.
        """
        quality = self.analyze(gray)

        return {
            "blur_score": round(quality["blur_score"], 2),
            "brightness": round(quality["brightness"], 2),
            "contrast": round(quality["contrast"], 2)
        }