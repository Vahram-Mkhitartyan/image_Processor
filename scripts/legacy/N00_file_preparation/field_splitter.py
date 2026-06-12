class FieldSplitter:
    def __init__(self, settings):
        """Initialize the object and store configuration.
        
        Args:
            settings: Optional configuration dictionary used to override defaults.
        
        Returns:
            None.
        """
        self.settings = settings

    def split_into_row_bands(
        self,
        image,
        y_levels,
        min_row_height=20
    ):
        """Split a page image into row-band crops.
        
        Args:
            image: Input image array.
            y_levels: Detected horizontal y-coordinate levels.
            min_row_height: Minimum row height to keep.
        
        Returns:
            List of row-band crop tuples.
        """
        row_bands = []

        margin = self.settings.get("row_margin", 12)

        height, width = image.shape[:2]

        if len(y_levels) == 0:
            row_bands.append((0, height, image))
            return row_bands

        first_y = y_levels[0]

        if first_y >= min_row_height:
            crop_y1 = 0
            crop_y2 = min(first_y + margin, height)

            row = image[crop_y1:crop_y2, :]
            row_bands.append((crop_y1, crop_y2, row))

        for index in range(len(y_levels) - 1):
            y1 = y_levels[index]
            y2 = y_levels[index + 1]

            if y2 - y1 < min_row_height:
                continue

            crop_y1 = max(y1 - margin, 0)
            crop_y2 = min(y2 + margin, height)

            row = image[crop_y1:crop_y2, :]

            row_bands.append((crop_y1, crop_y2, row))

        last_y = y_levels[-1]

        if height - last_y >= min_row_height:
            crop_y1 = max(last_y - margin, 0)
            crop_y2 = height

            row = image[crop_y1:crop_y2, :]
            row_bands.append((crop_y1, crop_y2, row))

        return row_bands

    def split_rows_into_fields(
        self,
        image,
        row_bands,
        vertical_lines,
        min_field_width=80
    ):
        """Split row bands into field-level crops.
        
        Args:
            image: Input image array.
            row_bands: List of row crop metadata tuples.
            vertical_lines: Detected vertical line boxes.
            min_field_width: Minimum field width to keep.
        
        Returns:
            List of field crop dictionaries.
        """
        field_crops = []

        margin = self.settings.get("field_margin", 8)

        image_height, image_width = image.shape[:2]

        for row_index, (row_y1, row_y2, row_image) in enumerate(row_bands, start=1):
            split_x_positions = [0, image_width]

            for x, y, w, h in vertical_lines:
                line_x = x + w // 2
                line_y1 = y
                line_y2 = y + h

                if line_y1 <= row_y2 and line_y2 >= row_y1:
                    split_x_positions.append(line_x)

            split_x_positions = sorted(set(split_x_positions))

            for field_index in range(len(split_x_positions) - 1):
                x1 = split_x_positions[field_index]
                x2 = split_x_positions[field_index + 1]

                if x2 - x1 < min_field_width:
                    continue

                crop_x1 = max(x1 - margin, 0)
                crop_x2 = min(x2 + margin, image_width)

                field_image = image[row_y1:row_y2, crop_x1:crop_x2]

                field_crops.append({
                    "row_index": row_index,
                    "field_index": field_index + 1,
                    "x1": crop_x1,
                    "y1": row_y1,
                    "x2": crop_x2,
                    "y2": row_y2,
                    "image": field_image
                })

        return field_crops