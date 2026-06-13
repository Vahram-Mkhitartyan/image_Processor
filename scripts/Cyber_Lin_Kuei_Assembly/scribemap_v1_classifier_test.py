import os
import sys
import json
import shutil
import cv2
import numpy as np
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
FILE_PREPARATION_DIR = f"{SCRIPTS_DIR}/N00_file_preparation"
SCRIBEMAP_DIR = f"{SCRIPTS_DIR}/N01_scribemap"

for path in [SCRIPTS_DIR, FILE_PREPARATION_DIR, SCRIBEMAP_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from N01_scribemap.scribemap_detector import ScribeMapBWDetector
from N00_file_preparation.file_preparation import prepare_file


class ScribeMapV1ClassifierTest:
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

        self.model_path = self.settings.get(
            "model_path",
            "/home/vahram/Desktop/image_Processor/models/handwriting_presence_v1_4.keras"
        )

        self.class_names = self.settings.get(
            "class_names",
            ["handwriting_present", "printed_only", "empty_or_noise"]
        )

        self.image_size = self.settings.get("image_size", (128, 128))
        self.debug_predictions = self.settings.get("debug_predictions", False)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")

        self.model = tf.keras.models.load_model(self.model_path)

        print("Loaded model:", self.model_path)
        print("Model input shape:", self.model.input_shape)
        print("Class names:", self.class_names)

    # ------------------------------------------------------------
    # OUTPUT FOLDERS
    # ------------------------------------------------------------

    def ensure_output_folders(self, output_dir):
        """Create and return the output folder structure.
        
        Args:
            output_dir: Folder where generated files should be written.
        
        Returns:
            Dictionary of created folder paths.
        """
        os.makedirs(output_dir, exist_ok=True)

        folders = {
            "scribemap_output": f"{output_dir}/scribemap_output",
            "classified": f"{output_dir}/classified",
            "handwriting_present": f"{output_dir}/classified/handwriting_present",
            "printed_only": f"{output_dir}/classified/printed_only",
            "empty_or_noise": f"{output_dir}/classified/empty_or_noise",
            "review": f"{output_dir}/classified/review",
            "metadata": f"{output_dir}/metadata"
        }

        for folder in folders.values():
            os.makedirs(folder, exist_ok=True)

        return folders

    # ------------------------------------------------------------
    # MODEL PREPROCESSING
    # ------------------------------------------------------------

    def resize_with_padding(self, image, target_size=128):
        """Resize an image into a padded square canvas.
        
        Args:
            image: Input image array.
            target_size: Desired square image size in pixels.
        
        Returns:
            Padded square image array.
        """
        height, width = image.shape[:2]

        if height <= 0 or width <= 0:
            raise ValueError("Invalid image size for padding.")

        scale = target_size / max(height, width)

        new_width = max(int(width * scale), 1)
        new_height = max(int(height * scale), 1)

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        canvas = 255 * np.ones(
            (target_size, target_size),
            dtype=np.uint8
        )

        x_offset = (target_size - new_width) // 2
        y_offset = (target_size - new_height) // 2

        canvas[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width
        ] = resized

        return canvas

    def preprocess_crop_for_model(self, image_path):
        """Load and prepare a crop for classifier inference.
        
        Args:
            image_path: Path to the image file.
        
        Returns:
            Prepared model input batch array.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Could not load crop: {image_path}")

        target_size = self.image_size[0]

        padded = self.resize_with_padding(
            image,
            target_size=target_size
        )

        normalized = padded.astype("float32") / 255.0

        # Model was trained on grayscale with channel dimension.
        normalized = np.expand_dims(normalized, axis=-1)

        batch = np.expand_dims(normalized, axis=0)

        return batch

    # ------------------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------------------

    def classify_crop(self, crop_path):
        """Classify one ScribeMap crop.
        
        Args:
            crop_path: Path to a crop image.
        
        Returns:
            Classification result dictionary.
        """
        if not os.path.exists(crop_path):
            raise FileNotFoundError(f"Crop path does not exist: {crop_path}")

        batch = self.preprocess_crop_for_model(crop_path)

        prediction = self.model.predict(batch, verbose=0)[0]

        predicted_index = int(np.argmax(prediction))
        label = self.class_names[predicted_index]
        confidence = float(prediction[predicted_index])

        class_scores = {
            self.class_names[index]: float(score)
            for index, score in enumerate(prediction)
        }

        handwriting_score = class_scores.get("handwriting_present", 0.0)

        contains_handwriting = handwriting_score >= self.settings.get(
            "handwriting_score_threshold",
            0.55
        )

        result = {
            "label": label,
            "confidence": confidence,
            "handwriting_score": handwriting_score,
            "contains_handwriting": contains_handwriting,
            "class_scores": class_scores
        }

        if self.debug_predictions:
            print("Prediction for:", crop_path)
            print(result)

        return result

    def copy_to_class_folder(self, crop_path, output_folders, result):
        """Copy a crop into the folder for its predicted class.
        
        Args:
            crop_path: Path to a crop image.
            output_folders: Dictionary of classifier output folder paths.
            result: Classification result dictionary.
        
        Returns:
            Path to the copied classified crop.
        """
        label = result["label"]

        if result["contains_handwriting"]:
            target_folder = output_folders["handwriting_present"]
        elif label == "printed_only":
            target_folder = output_folders["printed_only"]
        elif label == "empty_or_noise":
            target_folder = output_folders["empty_or_noise"]
        else:
            target_folder = output_folders["review"]

        file_name = os.path.basename(crop_path)
        target_path = f"{target_folder}/{file_name}"

        shutil.copy2(crop_path, target_path)

        return target_path

    # ------------------------------------------------------------
    # ONE DOCUMENT
    # ------------------------------------------------------------

    def run_one_document(self, input_bw_image_path, output_dir, document_id):
        """Run the ScribeMap classifier workflow for one document.
        
        Args:
            input_bw_image_path: Path to the prepared black/white input image.
            output_dir: Folder where generated files should be written.
            document_id: Stable identifier derived from the document filename.
        
        Returns:
            Summary dictionary for one document.
        """
        if not os.path.exists(input_bw_image_path):
            raise FileNotFoundError(f"Input BW image does not exist: {input_bw_image_path}")

        output_folders = self.ensure_output_folders(output_dir)

        scribemap_document_output = (
            f"{output_folders['scribemap_output']}/{document_id}"
        )

        preparation_steps = [
            "load_image",
            "rotate_major",
            "convert_to_grayscale",
            "denoise_image",
            "threshold_image",
            "deskew_image",
            "crop_white_margins",
            "create_scribemap_masks",
            "save_outputs",
        ]

        preparation_state = prepare_file(
            input_path=input_bw_image_path,
            output_dir=f"{scribemap_document_output}/preparation",
            steps=preparation_steps,
            settings=self.settings.get("file_preparation_settings", {})
        )

        scribemap = ScribeMapBWDetector(
            settings=self.settings.get("scribemap_settings", {})
        )

        scribemap_result = scribemap.run_from_preparation_state(
            preparation_state=preparation_state,
            output_dir=scribemap_document_output
        )

        print("ScribeMap output:", scribemap_document_output)
        print("ScribeMap groups:", scribemap_result.get("group_count"))

        if "groups" not in scribemap_result:
            raise KeyError("ScribeMap result has no 'groups' key.")

        if len(scribemap_result["groups"]) == 0:
            print("WARNING: ScribeMap produced 0 groups. Nothing to classify.")

        classified_groups = []

        for group in scribemap_result["groups"]:
            if "crop_path" not in group:
                raise KeyError(f"Group has no crop_path. Group sample: {group}")

            crop_path = group["crop_path"]

            if not os.path.exists(crop_path):
                raise FileNotFoundError(f"Group crop does not exist: {crop_path}")

            classification = self.classify_crop(crop_path)

            copied_path = self.copy_to_class_folder(
                crop_path,
                output_folders,
                classification
            )

            classified_group = {
                "document_id": document_id,
                "group_id": group["group_id"],
                "source_crop_path": crop_path,
                "classified_crop_path": copied_path,
                "bbox": {
                    "x1": group["x1"],
                    "y1": group["y1"],
                    "x2": group["x2"],
                    "y2": group["y2"]
                },
                "crop_bbox": group.get("crop_bbox"),
                "component_count": group["component_count"],
                "density": group["density"],
                "aspect_ratio": group["aspect_ratio"],
                "group_flags": group["group_flags"],
                "classification": classification
            }

            classified_groups.append(classified_group)

        handwriting_count = sum(
            1 for item in classified_groups
            if item["classification"]["contains_handwriting"]
        )

        printed_count = sum(
            1 for item in classified_groups
            if item["classification"]["label"] == "printed_only"
            and not item["classification"]["contains_handwriting"]
        )

        empty_or_noise_count = sum(
            1 for item in classified_groups
            if item["classification"]["label"] == "empty_or_noise"
            and not item["classification"]["contains_handwriting"]
        )

        review_count = len(classified_groups) - handwriting_count - printed_count - empty_or_noise_count

        summary = {
            "document_id": document_id,
            "input_bw_image_path": input_bw_image_path,
            "scribemap_group_count": scribemap_result["group_count"],
            "classified_group_count": len(classified_groups),
            "handwriting_count": handwriting_count,
            "printed_count": printed_count,
            "empty_or_noise_count": empty_or_noise_count,
            "review_count": review_count,
            "classified_groups": classified_groups,
            "scribemap_result_path": scribemap_result["metadata_path"],
            "scribemap_artifacts": scribemap_result["artifacts"]
        }

        summary_path = (
            f"{output_folders['metadata']}/{document_id}_classified_groups.json"
        )

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=4, ensure_ascii=False)

        summary["summary_path"] = summary_path

        return summary

    # ------------------------------------------------------------
    # BATCH
    # ------------------------------------------------------------

    def run_batch(self, input_paths, output_dir):
        """Run the ScribeMap classifier workflow for many documents.
        
        Args:
            input_paths: List of input document paths.
            output_dir: Folder where generated files should be written.
        
        Returns:
            Batch summary or None depending on caller context.
        """
        batch_results = []

        for index, input_path in enumerate(input_paths, start=1):
            document_id = f"doc_{index:03d}"

            print("=" * 80)
            print(f"Processing {document_id}: {input_path}")

            result = self.run_one_document(
                input_bw_image_path=input_path,
                output_dir=output_dir,
                document_id=document_id
            )

            print(
                f"{document_id}: groups={result['scribemap_group_count']} "
                f"classified={result['classified_group_count']} "
                f"handwriting={result['handwriting_count']} "
                f"printed={result['printed_count']} "
                f"noise={result['empty_or_noise_count']} "
                f"review={result['review_count']}"
            )

            batch_results.append(result)

        batch_summary = {
            "document_count": len(batch_results),
            "total_groups": sum(item["scribemap_group_count"] for item in batch_results),
            "total_classified": sum(item["classified_group_count"] for item in batch_results),
            "total_handwriting": sum(item["handwriting_count"] for item in batch_results),
            "total_printed": sum(item["printed_count"] for item in batch_results),
            "total_empty_or_noise": sum(item["empty_or_noise_count"] for item in batch_results),
            "total_review": sum(item["review_count"] for item in batch_results),
            "documents": [
                {
                    "document_id": item["document_id"],
                    "input_bw_image_path": item["input_bw_image_path"],
                    "summary_path": item["summary_path"],
                    "groups": item["scribemap_group_count"],
                    "classified": item["classified_group_count"],
                    "handwriting": item["handwriting_count"],
                    "printed": item["printed_count"],
                    "empty_or_noise": item["empty_or_noise_count"],
                    "review": item["review_count"]
                }
                for item in batch_results
            ]
        }

        metadata_dir = f"{output_dir}/metadata"
        os.makedirs(metadata_dir, exist_ok=True)

        batch_summary_path = f"{metadata_dir}/batch_summary.json"

        with open(batch_summary_path, "w", encoding="utf-8") as file:
            json.dump(batch_summary, file, indent=4, ensure_ascii=False)

        print("=" * 80)
        print("Batch complete.")
        print("Batch summary:", batch_summary_path)

        return batch_summary


if __name__ == "__main__":
    BASE_DIR = "/home/vahram/Desktop/image_Processor"

    input_paths = [
        f"{BASE_DIR}/handwritten_text/test_bw_1.jpeg",
        f"{BASE_DIR}/handwritten_text/test_bw_2.jpeg",
        f"{BASE_DIR}/handwritten_text/test_bw_3.jpeg"
    ]

    output_dir = f"{BASE_DIR}/scribemap_v1_test_output"

    tester = ScribeMapV1ClassifierTest(
        settings={
            "model_path": f"{BASE_DIR}/models/handwriting_presence_v1_4.keras",
            "class_names": ["handwriting_present", "printed_only", "empty_or_noise"],
            "image_size": (128, 128),

            # High recall mode.
            "handwriting_score_threshold": 0.35,

            # Set True only for a small test because it prints every prediction.
            "debug_predictions": False,

            "file_preparation_settings": {
                "remove_grouped_vertical_lines": True,
                "remove_vertical_lines": False,
                "vertical_cluster_min_fragments": 2,
                "vertical_cluster_min_total_height": 60,
                "vertical_cluster_min_y_span": 80,
                "grouped_vertical_removal_half_width": 3,
            },

            "scribemap_settings": {
                "black_pixel_threshold": 80,

                "horizontal_line_kernel": (80, 1),
                "short_horizontal_line_kernel": (25, 1),
                "vertical_line_kernel": (1, 40),

                # Kept for compatibility with manual classifier checks.
                "remove_vertical_lines": False,
                "remove_grouped_vertical_lines": True,

                "vertical_fragment_min_height": 10,
                "vertical_fragment_max_width": 20,
                "vertical_fragment_min_aspect_ratio": 2.5,
                "vertical_cluster_x_tolerance": 10,
                "vertical_cluster_min_fragments": 2,
                "vertical_cluster_min_total_height": 60,
                "vertical_cluster_min_y_span": 80,
                "grouped_vertical_removal_half_width": 3,

                "min_component_area": 15,
                "max_component_area": 999000,
                "min_component_width": 1,
                "min_component_height": 1,

                "wide_line_aspect_ratio": 20,
                "vertical_line_aspect_ratio": 0.2,

                "component_margin": 4,
                "group_crop_margin": 10,

                "group_comparison_y_window": 64,
                "group_max_horizontal_gap": 45,
                "group_y_tolerance": 22,
                "group_min_vertical_overlap": 0.18,
                "group_max_height_ratio": 3.7,
                "stage_a_max_pair_merge_width": 370,
                "stage_a_max_pair_merge_height": 82,
                "stage_a_max_group_width": 295,
                "stage_a_max_group_height": 98,
                "group_ignore_wide_line_like": True,
                "min_group_components": 1,
                "group_wide_aspect_ratio": 20,
                "group_vertical_aspect_ratio": 0.2,
                "reject_horizontal_min_width": 18,
                "reject_horizontal_max_height": 10,
                "reject_horizontal_min_aspect": 5,
                "reject_vertical_min_height": 18,
                "reject_vertical_max_width": 10,
                "reject_vertical_min_aspect": 4,
                "reject_tiny_group_max_area": 60,
                "reject_tiny_group_max_width": 10,
                "reject_tiny_group_max_height": 10,
                "reject_dense_blob_max_area": 200,
                "reject_dense_blob_min_density": 0.80,
                "reject_oversized_group_max_width": 350,
                "reject_oversized_group_max_height": 122,
                "reject_oversized_group_max_area": 34000
            }
        }
    )

    tester.run_batch(
        input_paths=input_paths,
        output_dir=output_dir
    )
