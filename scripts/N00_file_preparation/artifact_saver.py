import os
import json
import cv2


class ArtifactSaver:
    def __init__(self, output_dir):
        """Initialize the object and store configuration.
        
        Args:
            output_dir: Folder where generated files should be written.
        
        Returns:
            None.
        """
        self.output_dir = output_dir

    def ensure_output_folders(self):
        """Create and return the output folder structure.
        
        Args:
            None.
        
        Returns:
            Dictionary of created folder paths.
        """
        full_images_dir = f"{self.output_dir}/full_images"
        lines_dir = f"{self.output_dir}/lines"
        masks_dir = f"{self.output_dir}/masks"
        metadata_dir = f"{self.output_dir}/metadata"

        os.makedirs(full_images_dir, exist_ok=True)
        os.makedirs(lines_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)

        return {
            "full_images_dir": full_images_dir,
            "lines_dir": lines_dir,
            "masks_dir": masks_dir,
            "metadata_dir": metadata_dir
        }

    def save_stage(self, image, stage_name, folder_path, extension=".jpeg"):
        """Save an image stage to disk.
        
        Args:
            image: Input image array.
            stage_name: Base name for the saved processing stage.
            folder_path: Folder where the file should be saved.
            extension: File extension to use when saving the image.
        
        Returns:
            Path to the saved image file.
        """
        output_path = f"{folder_path}/{stage_name}{extension}"

        success = cv2.imwrite(output_path, image)

        if not success:
            raise ValueError(f"Could not save image to path: {output_path}")

        return output_path

    def save_mask_stage(self, image, stage_name, folder_path):
        # Masks are consumed downstream, so they must stay binary/lossless.
        """Save a binary mask losslessly for downstream processing.
        
        Args:
            image: Input image array.
            stage_name: Base name for the saved processing stage.
            folder_path: Folder where the file should be saved.
        
        Returns:
            Path to the saved PNG mask file.
        """
        return self.save_stage(image, stage_name, folder_path, extension=".png")

    def save_json(self, data, output_path):
        """Write JSON data to disk.
        
        Args:
            data: Serializable data object.
            output_path: Path where the result should be saved.
        
        Returns:
            Path to the saved JSON file.
        """
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

        return output_path

    def save_outputs(self, state):
        """Persist available pipeline artifacts and metadata paths.
        
        Args:
            state: Mutable pipeline state dictionary.
        
        Returns:
            Updated pipeline state with artifact paths.
        """
        folders = self.ensure_output_folders()

        state["artifacts"]["folders"] = folders

        # Remove files from the retired enhancement-era sequence so repeated
        # prep runs cannot expose stale stages in metadata or the UI.
        for retired_name in (
            "04_enhanced.jpeg",
            "05_thresholded.jpeg",
            "06_deskewed.jpeg",
            "07_cropped.jpeg",
        ):
            retired_path = os.path.join(
                folders["full_images_dir"],
                retired_name,
            )
            if os.path.isfile(retired_path):
                os.remove(retired_path)

        if "gray" in state["images"]:
            state["artifacts"]["gray"] = self.save_stage(
                state["images"]["gray"],
                "02_gray",
                folders["full_images_dir"]
            )

        if "denoised" in state["images"]:
            state["artifacts"]["denoised"] = self.save_stage(
                state["images"]["denoised"],
                "03_denoised",
                folders["full_images_dir"]
            )

        if "thresholded" in state["images"]:
            state["artifacts"]["thresholded"] = self.save_stage(
                state["images"]["thresholded"],
                "04_thresholded",
                folders["full_images_dir"]
            )

        if "deskewed" in state["images"]:
            state["artifacts"]["deskewed"] = self.save_stage(
                state["images"]["deskewed"],
                "05_deskewed",
                folders["full_images_dir"]
            )

        if "cropped" in state["images"]:
            state["artifacts"]["cropped"] = self.save_stage(
                state["images"]["cropped"],
                "06_cropped",
                folders["full_images_dir"]
            )

        if "horizontal_lines_mask" in state["images"]:
            state["artifacts"]["horizontal_lines_mask"] = self.save_mask_stage(
                state["images"]["horizontal_lines_mask"],
                "horizontal_mask",
                folders["lines_dir"]
            )

        if "horizontal_lines_preview" in state["images"]:
            state["artifacts"]["horizontal_lines_preview"] = self.save_stage(
                state["images"]["horizontal_lines_preview"],
                "horizontal_preview",
                folders["lines_dir"]
            )

        if "vertical_lines_mask" in state["images"]:
            state["artifacts"]["vertical_lines_mask"] = self.save_mask_stage(
                state["images"]["vertical_lines_mask"],
                "vertical_mask",
                folders["lines_dir"]
            )

        if "vertical_lines_preview" in state["images"]:
            state["artifacts"]["vertical_lines_preview"] = self.save_stage(
                state["images"]["vertical_lines_preview"],
                "vertical_preview",
                folders["lines_dir"]
            )

        if "combined_lines_preview" in state["images"]:
            state["artifacts"]["combined_lines_preview"] = self.save_stage(
                state["images"]["combined_lines_preview"],
                "combined_preview",
                folders["lines_dir"]
            )

        if "black_pixel_mask" in state["images"]:
            state["artifacts"]["black_pixel_mask"] = self.save_mask_stage(
                state["images"]["black_pixel_mask"],
                "01_black_pixel_mask",
                folders["masks_dir"]
            )

        if "horizontal_line_mask" in state["images"]:
            state["artifacts"]["horizontal_line_mask"] = self.save_mask_stage(
                state["images"]["horizontal_line_mask"],
                "02_horizontal_line_mask",
                folders["masks_dir"]
            )

        if "short_horizontal_line_mask" in state["images"]:
            state["artifacts"]["short_horizontal_line_mask"] = self.save_mask_stage(
                state["images"]["short_horizontal_line_mask"],
                "03_short_horizontal_line_mask",
                folders["masks_dir"]
            )

        if "combined_horizontal_line_mask" in state["images"]:
            state["artifacts"]["combined_horizontal_line_mask"] = self.save_mask_stage(
                state["images"]["combined_horizontal_line_mask"],
                "04_combined_horizontal_line_mask",
                folders["masks_dir"]
            )

        if "vertical_line_mask" in state["images"]:
            state["artifacts"]["vertical_line_mask"] = self.save_mask_stage(
                state["images"]["vertical_line_mask"],
                "05_vertical_line_mask",
                folders["masks_dir"]
            )

        if "grouped_vertical_line_mask" in state["images"]:
            state["artifacts"]["grouped_vertical_line_mask"] = self.save_mask_stage(
                state["images"]["grouped_vertical_line_mask"],
                "06_grouped_vertical_line_mask",
                folders["masks_dir"]
            )

        if "content_ink_mask" in state["images"]:
            state["artifacts"]["content_ink_mask"] = self.save_mask_stage(
                state["images"]["content_ink_mask"],
                "07_content_ink_mask",
                folders["masks_dir"]
            )
        
        if "red_ink_mask" in state["images"]:
            state["artifacts"]["red_ink_mask"] = self.save_mask_stage(
                state["images"]["red_ink_mask"],
                "08_red_ink_mask",
                folders["masks_dir"]
            )

        if "blue_ink_mask" in state["images"]:
            state["artifacts"]["blue_ink_mask"] = self.save_mask_stage(
                state["images"]["blue_ink_mask"],
                "09_blue_ink_mask",
                folders["masks_dir"]
            )

        if "green_ink_mask" in state["images"]:
            state["artifacts"]["green_ink_mask"] = self.save_mask_stage(
                state["images"]["green_ink_mask"],
                "10_green_ink_mask",
                folders["masks_dir"]
            )

        if "unknown_color_ink_mask" in state["images"]:
            state["artifacts"]["unknown_color_ink_mask"] = self.save_mask_stage(
                state["images"]["unknown_color_ink_mask"],
                "11_unknown_color_ink_mask",
                folders["masks_dir"]
            )

        if "colored_ink_mask" in state["images"]:
            state["artifacts"]["colored_ink_mask"] = self.save_mask_stage(
                state["images"]["colored_ink_mask"],
                "12_colored_ink_mask",
                folders["masks_dir"]
            )

        if "black_ink_mask" in state["images"]:
            state["artifacts"]["black_ink_mask"] = self.save_mask_stage(
                state["images"]["black_ink_mask"],
                "13_black_ink_mask",
                folders["masks_dir"]
            )

        if "blue_continuity_mask" in state["images"]:
            state["artifacts"]["blue_continuity_mask"] = self.save_mask_stage(
                state["images"]["blue_continuity_mask"],
                "14_blue_continuity_mask",
                folders["masks_dir"]
            )

        if "red_continuity_mask" in state["images"]:
            state["artifacts"]["red_continuity_mask"] = self.save_mask_stage(
                state["images"]["red_continuity_mask"],
                "15_red_continuity_mask",
                folders["masks_dir"]
            )

        if "blue_borrowed_bridge_mask" in state["images"]:
            state["artifacts"]["blue_borrowed_bridge_mask"] = self.save_mask_stage(
                state["images"]["blue_borrowed_bridge_mask"],
                "16_blue_borrowed_bridge_mask",
                folders["masks_dir"]
            )

        if "red_borrowed_bridge_mask" in state["images"]:
            state["artifacts"]["red_borrowed_bridge_mask"] = self.save_mask_stage(
                state["images"]["red_borrowed_bridge_mask"],
                "17_red_borrowed_bridge_mask",
                folders["masks_dir"]
            )


        if "red_ink_layer" in state["images"]:
            state["artifacts"]["red_ink_layer"] = self.save_stage(
                state["images"]["red_ink_layer"],
                "08_red_ink_layer",
                folders["full_images_dir"]
            )

        if "blue_ink_layer" in state["images"]:
            state["artifacts"]["blue_ink_layer"] = self.save_stage(
                state["images"]["blue_ink_layer"],
                "09_blue_ink_layer",
                folders["full_images_dir"]
            )

        if "green_ink_layer" in state["images"]:
            state["artifacts"]["green_ink_layer"] = self.save_stage(
                state["images"]["green_ink_layer"],
                "10_green_ink_layer",
                folders["full_images_dir"]
            )

        if "unknown_color_ink_layer" in state["images"]:
            state["artifacts"]["unknown_color_ink_layer"] = self.save_stage(
                state["images"]["unknown_color_ink_layer"],
                "11_unknown_color_ink_layer",
                folders["full_images_dir"]
            )

        if "colored_ink_layer" in state["images"]:
            state["artifacts"]["colored_ink_layer"] = self.save_stage(
                state["images"]["colored_ink_layer"],
                "12_colored_ink_layer",
                folders["full_images_dir"]
            )

        if "black_ink_layer" in state["images"]:
            state["artifacts"]["black_ink_layer"] = self.save_stage(
                state["images"]["black_ink_layer"],
                "13_black_ink_layer",
                folders["full_images_dir"]
            )

        metadata_path = f"{folders['metadata_dir']}/metadata.json"

        state["artifacts"]["metadata"] = self.save_json(
            state["metadata"],
            metadata_path
        )

        return state
