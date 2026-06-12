"""Visual debug rendering for ScribeTrace intermediate geometry."""

import os

import cv2

from .trace_settings import normalize_trace_settings

class TraceDebugWriter:
    """Write component, skeleton, graph, and ordered-path debug images."""

    def __init__(self, settings=None):
        self.settings = normalize_trace_settings(settings)

    def ensure_output_dir(self, output_path):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def load_debug_base_image(self, image_path):
        grayscale = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if grayscale is None:
            raise ValueError(f"Failed to load debug source image: {image_path}")
        return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

    def draw_component_boxes(self, image, components):
        debug_image = image.copy()
        for component in components:
            box = component.bounding_box
            cv2.rectangle(
                debug_image,
                (box.x1, box.y1),
                (box.x2 - 1, box.y2 - 1),
                (0, 255, 0),
                1,
            )
            if self.settings.debug_draw_labels:
                label_y = box.y1 - 3 if box.y1 >= 10 else box.y1 + 10
                cv2.putText(
                    debug_image,
                    str(component.component_id),
                    (box.x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        return debug_image

    def save_component_debug(self, image_path, components, output_path):
        self.ensure_output_dir(output_path)
        image = self.draw_component_boxes(
            self.load_debug_base_image(image_path), components
        )
        if not cv2.imwrite(output_path, image):
            raise ValueError(f"Failed to save component debug image: {output_path}")
        return output_path

    def save_skeleton_debug(self, skeleton_mask, output_path):
        self.ensure_output_dir(output_path)
        if not cv2.imwrite(output_path, skeleton_mask):
            raise ValueError(f"Failed to save skeleton debug image: {output_path}")
        return output_path

    def draw_skeleton_graph_overlay(self, skeleton_mask, skeleton_graph):
        image = cv2.cvtColor(skeleton_mask, cv2.COLOR_GRAY2BGR)
        for point in skeleton_graph.endpoints():
            cv2.circle(image, point.to_tuple(), 2, (0, 0, 255), -1)
        for point in skeleton_graph.junction_cluster_centers():
            cv2.circle(image, point.to_tuple(), 2, (0, 255, 255), -1)
        for point in skeleton_graph.isolated_points():
            cv2.circle(image, point.to_tuple(), 2, (255, 0, 0), -1)
        return image

    def save_skeleton_graph_debug(self, skeleton_mask, graph, output_path):
        self.ensure_output_dir(output_path)
        image = self.draw_skeleton_graph_overlay(skeleton_mask, graph)
        if not cv2.imwrite(output_path, image):
            raise ValueError(f"Failed to save graph debug image: {output_path}")
        return output_path

    def draw_trace_paths_overlay(self, skeleton_mask, trace_paths):
        image = cv2.cvtColor(skeleton_mask, cv2.COLOR_GRAY2BGR)
        for path in trace_paths:
            color = (255, 200, 0) if path.is_closed else (0, 255, 0)
            for point_a, point_b in zip(path.points, path.points[1:]):
                cv2.line(image, point_a.to_tuple(), point_b.to_tuple(), color, 1)
            if self.settings.debug_draw_labels:
                start = path.start_point()
                cv2.putText(
                    image,
                    str(path.path_id),
                    (start.x + 2, start.y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            if path.is_closed and path.points[0].to_tuple() != path.points[-1].to_tuple():
                cv2.line(
                    image,
                    path.points[-1].to_tuple(),
                    path.points[0].to_tuple(),
                    color,
                    1,
                )
        return image

    def save_trace_paths_debug(self, skeleton_mask, trace_paths, output_path):
        self.ensure_output_dir(output_path)
        image = self.draw_trace_paths_overlay(skeleton_mask, trace_paths)
        if not cv2.imwrite(output_path, image):
            raise ValueError(f"Failed to save path debug image: {output_path}")
        return output_path
    
    def draw_landmarks_overlay(self, skeleton_mask, trace_paths, landmarks):
        image = self.draw_trace_paths_overlay(
            skeleton_mask=skeleton_mask,
            trace_paths=trace_paths,
        )

        color_by_type = {
            "start": (0, 0, 255),
            "end": (255, 0, 0),
            "global_left": (255, 255, 0),
            "global_right": (255, 255, 0),
            "global_top": (255, 0, 255),
            "global_bottom": (255, 0, 255),
            "local_top_peak": (0, 255, 255),
            "local_bottom_valley": (0, 128, 255),
            "local_left_turn": (255, 128, 0),
            "local_right_turn": (255, 128, 0),
        }

        label_by_type = {
            "start": "S",
            "end": "E",
            "global_left": "GL",
            "global_right": "GR",
            "global_top": "GT",
            "global_bottom": "GB",
            "local_top_peak": "TP",
            "local_bottom_valley": "BV",
            "local_left_turn": "LT",
            "local_right_turn": "RT",
        }

        for landmark in landmarks:
            point = landmark.point
            color = color_by_type.get(landmark.landmark_type, (0, 255, 255))
            label = label_by_type.get(landmark.landmark_type, "LM")

            cv2.circle(image, point.to_tuple(), 2, color, -1)

            if self.settings.debug_draw_labels:
                cv2.putText(
                    image,
                    label,
                    (point.x + 2, point.y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        return image


    def save_landmarks_debug(self, skeleton_mask, trace_paths, landmarks, output_path):
        self.ensure_output_dir(output_path)

        image = self.draw_landmarks_overlay(
            skeleton_mask=skeleton_mask,
            trace_paths=trace_paths,
            landmarks=landmarks,
        )

        if not cv2.imwrite(output_path, image):
            raise ValueError(f"Failed to save landmark debug image: {output_path}")

        return output_path



