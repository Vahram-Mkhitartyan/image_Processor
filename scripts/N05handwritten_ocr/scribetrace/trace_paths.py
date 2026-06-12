"""Ordered skeleton paths, conservative spur merging, and landmarks."""

import math
from collections import deque

from .trace_common import coordinate_key, edge_key
from .trace_models import SkeletonPoint, TraceLandmark, TracePath
from .trace_settings import normalize_trace_settings

_coordinate_key = coordinate_key

class PathSignal:
    """
    Treat an ordered TracePath as coordinate signals x(t), y(t).

    This converts geometry into a sequence-analysis problem:
    - x(t): horizontal motion
    - y(t): vertical motion

    Since image coordinates grow downward:
    - local y minimum = visual top peak
    - local y maximum = visual bottom valley
    """

    def __init__(self, trace_path):
        self.trace_path = trace_path
        self.points = list(trace_path.points)
        self.x_values = [point.x for point in self.points]
        self.y_values = [point.y for point in self.points]

    def point_count(self):
        return len(self.points)

    def value_at(self, axis, index):
        if axis == "x":
            return self.x_values[index]
        if axis == "y":
            return self.y_values[index]
        raise ValueError(f"Unsupported signal axis: {axis}")

    def local_extrema_indices(
        self,
        axis,
        min_prominence=2,
        min_spacing=3,
    ):
        """
        Detect local extrema in x(t) or y(t).

        v0_2 fix:
        The v0_1 method compared a turning point against the immediate next
        sample. For one-pixel skeleton walks, that made the right-side
        prominence almost always 1, so features with min_prominence=2 became
        dead. This version first compresses the signal into monotonic runs and
        measures the turn against the full left and right runs.
        """
        values = self.x_values if axis == "x" else self.y_values

        if len(values) < 3:
            return []

        runs = []
        current_trend = 0
        run_start = 0
        previous_nonflat_index = 0

        for index in range(1, len(values)):
            delta = values[index] - values[index - 1]
            if delta > 0:
                trend = 1
            elif delta < 0:
                trend = -1
            else:
                continue

            if current_trend == 0:
                current_trend = trend
                run_start = index - 1
                previous_nonflat_index = index
                continue

            if trend != current_trend:
                runs.append(
                    {
                        "start": run_start,
                        "end": index - 1,
                        "trend": current_trend,
                    }
                )
                run_start = index - 1
                current_trend = trend

            previous_nonflat_index = index

        if current_trend != 0:
            runs.append(
                {
                    "start": run_start,
                    "end": previous_nonflat_index,
                    "trend": current_trend,
                }
            )

        if len(runs) < 2:
            return []

        candidates = []

        for left_run, right_run in zip(runs, runs[1:]):
            if left_run["trend"] == right_run["trend"]:
                continue

            extremum_index = left_run["end"]
            extremum_value = values[extremum_index]
            left_value = values[left_run["start"]]
            right_value = values[right_run["end"]]

            prominence = min(
                abs(extremum_value - left_value),
                abs(extremum_value - right_value),
            )

            if prominence < min_prominence:
                continue

            if left_run["trend"] < 0 and right_run["trend"] > 0:
                extremum_kind = "min"
            else:
                extremum_kind = "max"

            candidates.append(
                {
                    "index": extremum_index,
                    "kind": extremum_kind,
                    "prominence": prominence,
                }
            )

        return self._filter_by_spacing(candidates, min_spacing)

    def _filter_by_spacing(self, candidates, min_spacing):
        """
        Suppress extrema that are too close.

        If two candidates are close, keep the one with higher prominence.
        Ties are deterministic by lower index.
        """
        accepted = []

        for candidate in candidates:
            too_close_index = None

            for index, accepted_candidate in enumerate(accepted):
                if abs(candidate["index"] - accepted_candidate["index"]) < min_spacing:
                    too_close_index = index
                    break

            if too_close_index is None:
                accepted.append(candidate)
                continue

            existing = accepted[too_close_index]

            if (
                candidate["prominence"] > existing["prominence"]
                or (
                    candidate["prominence"] == existing["prominence"]
                    and candidate["index"] < existing["index"]
                )
            ):
                accepted[too_close_index] = candidate

        accepted.sort(key=lambda item: item["index"])
        return accepted

class TracePathExtractor:
    """Extract every logical edge and conservatively merge terminal spurs."""

    def __init__(self, settings=None):
        self.settings = normalize_trace_settings(settings)
        self.metrics = {}

    @staticmethod
    def edge_key(point_a, point_b):
        return edge_key(point_a, point_b)

    def _node_metadata(self, graph, point):
        if graph.is_junction_point(point):
            return "junction", graph.junction_cluster_id_for_point(point)
        if graph.is_endpoint(point):
            return "endpoint", None
        if graph.neighbor_count(point) == 0:
            return "isolated", None
        return "path", None

    def _walk_segment(self, graph, start, first, visited_edges):
        points = [start, first]
        visited_edges.add(self.edge_key(start, first))
        previous = start
        current = first

        while True:
            if graph.is_endpoint(current) or graph.is_junction_point(current):
                break
            candidates = [
                neighbor
                for neighbor in graph.neighbors_of(current)
                if neighbor.to_tuple() != previous.to_tuple()
                and self.edge_key(current, neighbor) not in visited_edges
            ]
            if not candidates:
                break
            next_point = sorted(candidates, key=_coordinate_key)[0]
            visited_edges.add(self.edge_key(current, next_point))
            previous, current = current, next_point
            points.append(current)
        return points

    def _canonicalize_closed_points(self, points):
        """Rotate and orient a loop around its stable lexicographic anchor."""
        coordinates = [point.to_tuple() for point in points]
        if coordinates[0] == coordinates[-1]:
            coordinates = coordinates[:-1]
        if not coordinates:
            return points

        candidates = []
        for sequence in (coordinates, list(reversed(coordinates))):
            for index, coordinate in enumerate(sequence):
                if coordinate == min(sequence):
                    candidates.append(sequence[index:] + sequence[:index])
        canonical = min(candidates)
        canonical.append(canonical[0])
        return [SkeletonPoint(x, y) for x, y in canonical]

    def _walk_remaining_edge(self, graph, edge, visited_edges):
        start = graph.point_lookup[edge[0]]
        first = graph.point_lookup[edge[1]]
        points = [start, first]
        visited_edges.add(edge)
        previous = start
        current = first

        while True:
            candidates = [
                neighbor
                for neighbor in graph.neighbors_of(current)
                if neighbor.to_tuple() != previous.to_tuple()
                and self.edge_key(current, neighbor) not in visited_edges
            ]
            if not candidates:
                break
            next_point = sorted(candidates, key=_coordinate_key)[0]
            visited_edges.add(self.edge_key(current, next_point))
            previous, current = current, next_point
            points.append(current)
            if current.to_tuple() == start.to_tuple():
                break

        is_closed = points[-1].to_tuple() == points[0].to_tuple()
        if is_closed:
            points = self._canonicalize_closed_points(points)
        return points, is_closed

    def _path_sort_key(self, path):
        return (
            path.start_point().y,
            path.start_point().x,
            path.end_point().y,
            path.end_point().x,
            tuple(point.to_tuple() for point in path.points),
        )

    def _make_path(self, graph, path_id, points, is_closed=False):
        start_type, start_cluster = self._node_metadata(graph, points[0])
        end_type, end_cluster = self._node_metadata(graph, points[-1])

        same_coordinate_cycle = (
            len(points) >= 4
            and points[0].to_tuple() == points[-1].to_tuple()
        )

        same_junction_cycle = (
            len(points) >= 4
            and start_type == "junction"
            and end_type == "junction"
            and start_cluster is not None
            and start_cluster == end_cluster
        )

        if is_closed or same_coordinate_cycle or same_junction_cycle:
            is_closed = True

            # Pure leftover loops have no real endpoint/junction anchor.
            # Attached loops return to a junction, so keep that information.
            if same_junction_cycle:
                start_type = "attached_loop"
                end_type = "attached_loop"
            else:
                start_type = "loop"
                end_type = "loop"
                start_cluster = None
                end_cluster = None

        return TracePath(
            path_id=path_id,
            points=points,
            is_closed=is_closed,
            start_node_type=start_type,
            end_node_type=end_type,
            start_junction_cluster_id=start_cluster,
            end_junction_cluster_id=end_cluster,
            stop_reason="closed_loop" if is_closed else end_type,
        )

    def _junction_for_terminal_spur(self, path):
        start_is_junction = path.start_node_type == "junction"
        end_is_junction = path.end_node_type == "junction"
        start_is_endpoint = path.start_node_type == "endpoint"
        end_is_endpoint = path.end_node_type == "endpoint"
        if start_is_junction and end_is_endpoint:
            return path.start_junction_cluster_id
        if end_is_junction and start_is_endpoint:
            return path.end_junction_cluster_id
        return None

    def _oriented_from_junction(self, path, cluster_id):
        if path.start_junction_cluster_id == cluster_id:
            return list(path.points)
        if path.end_junction_cluster_id == cluster_id:
            return list(reversed(path.points))
        return None

    def _tangent_vector(self, path, cluster_id):
        points = self._oriented_from_junction(path, cluster_id)
        if not points or len(points) < 2:
            return None
        index = min(len(points) - 1, self.settings.short_path_tangent_points - 1)
        return (
            points[index].x - points[0].x,
            points[index].y - points[0].y,
        )

    @staticmethod
    def _continuation_deviation(vector_a, vector_b):
        magnitude_a = math.hypot(*vector_a)
        magnitude_b = math.hypot(*vector_b)
        if magnitude_a == 0 or magnitude_b == 0:
            return 180.0
        cosine = (
            vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
        ) / (magnitude_a * magnitude_b)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        return abs(180.0 - angle)

    def _cluster_bridge(self, graph, start, end, cluster_id):
        """Find a deterministic path through one contracted junction cluster."""
        if start.to_tuple() == end.to_tuple():
            return [start]
        allowed = {
            point.to_tuple()
            for point in graph.junction_clusters()[cluster_id]
        }
        queue = deque([start.to_tuple()])
        parents = {start.to_tuple(): None}
        while queue:
            coordinate = queue.popleft()
            current = graph.point_lookup[coordinate]
            for neighbor in graph.neighbors_of(current):
                next_coordinate = neighbor.to_tuple()
                if next_coordinate not in allowed or next_coordinate in parents:
                    continue
                parents[next_coordinate] = coordinate
                if next_coordinate == end.to_tuple():
                    queue.clear()
                    break
                queue.append(next_coordinate)
        if end.to_tuple() not in parents:
            return [start, end]
        coordinates = []
        current = end.to_tuple()
        while current is not None:
            coordinates.append(current)
            current = parents[current]
        coordinates.reverse()
        return [graph.point_lookup[coordinate] for coordinate in coordinates]

    def _merge_pair(self, graph, spur, target, cluster_id):
        spur_outward = self._oriented_from_junction(spur, cluster_id)
        target_outward = self._oriented_from_junction(target, cluster_id)
        bridge = self._cluster_bridge(
            graph, spur_outward[0], target_outward[0], cluster_id
        )
        points = (
            list(reversed(spur_outward))
            + bridge[1:]
            + target_outward[1:]
        )
        source_ids = []
        for path in (spur, target):
            source_ids.extend(path.merged_from_path_ids or [path.path_id])
        if target.start_junction_cluster_id == cluster_id:
            other_node_type = target.end_node_type
            other_cluster_id = target.end_junction_cluster_id
        else:
            other_node_type = target.start_node_type
            other_cluster_id = target.start_junction_cluster_id
        return TracePath(
            path_id=min(spur.path_id, target.path_id),
            points=points,
            start_node_type="endpoint",
            end_node_type=other_node_type,
            end_junction_cluster_id=other_cluster_id,
            stop_reason="directional_spur_merge",
            merged_from_path_ids=sorted(set(source_ids)),
        )

    def _merge_short_terminal_spurs(self, graph, paths):
        minimum_points = self.settings.minimum_trace_path_points
        for path in paths:
            path.is_short = path.point_count() < minimum_points

        consumed = set()
        merged_paths = []
        merge_count = 0
        for spur in sorted(paths, key=lambda item: item.path_id):
            if spur.path_id in consumed or not spur.is_short or spur.is_closed:
                continue
            cluster_id = self._junction_for_terminal_spur(spur)
            if cluster_id is None:
                continue
            spur_vector = self._tangent_vector(spur, cluster_id)
            if spur_vector is None:
                continue

            candidates = []
            for candidate in paths:
                if (
                    candidate.path_id == spur.path_id
                    or candidate.path_id in consumed
                    or candidate.is_closed
                    or candidate.is_short
                ):
                    continue
                candidate_points = self._oriented_from_junction(candidate, cluster_id)
                candidate_vector = self._tangent_vector(candidate, cluster_id)
                if candidate_points is None or candidate_vector is None:
                    continue
                deviation = self._continuation_deviation(
                    spur_vector, candidate_vector
                )
                candidates.append((deviation, candidate.path_id, candidate))

            candidates.sort(key=lambda item: (item[0], item[1]))
            if not candidates:
                continue
            best_deviation, _, target = candidates[0]
            next_deviation = candidates[1][0] if len(candidates) > 1 else math.inf
            advantage = next_deviation - best_deviation
            if (
                best_deviation > self.settings.short_path_merge_max_angle_degrees
                or advantage
                < self.settings.short_path_merge_min_advantage_degrees
            ):
                continue

            merged_paths.append(self._merge_pair(graph, spur, target, cluster_id))
            consumed.update((spur.path_id, target.path_id))
            merge_count += 1

        result = [
            path for path in paths
            if path.path_id not in consumed
        ] + merged_paths
        result.sort(key=self._path_sort_key)
        for path_id, path in enumerate(result):
            path.path_id = path_id
            path.is_short = path.point_count() < minimum_points
        return result, merge_count

    def extract_paths(self, skeleton_graph):
        """Traverse terminal segments and every remaining graph edge."""
        visited_edges = set()

        # Junction clusters are logical nodes, so their internal pixel edges do
        # not become fake trace paths.
        for cluster in skeleton_graph.junction_clusters():
            cluster_coordinates = {point.to_tuple() for point in cluster}
            for point in cluster:
                for neighbor in skeleton_graph.neighbors_of(point):
                    if neighbor.to_tuple() in cluster_coordinates:
                        visited_edges.add(self.edge_key(point, neighbor))

        raw_paths = []
        starts = []
        for endpoint in skeleton_graph.endpoints():
            for neighbor in skeleton_graph.neighbors_of(endpoint):
                starts.append((0, endpoint, neighbor))
        for cluster_id, cluster in enumerate(skeleton_graph.junction_clusters()):
            for junction in cluster:
                for neighbor in skeleton_graph.neighbors_of(junction):
                    if not skeleton_graph.is_junction_point(neighbor):
                        starts.append((cluster_id + 1, junction, neighbor))
        starts.sort(
            key=lambda item: (
                item[0],
                item[1].y,
                item[1].x,
                item[2].y,
                item[2].x,
            )
        )

        for _, start, first in starts:
            edge = self.edge_key(start, first)
            if edge in visited_edges:
                continue
            points = self._walk_segment(
                skeleton_graph, start, first, visited_edges
            )
            raw_paths.append(
                self._make_path(skeleton_graph, len(raw_paths), points)
            )

        for edge in skeleton_graph.all_edges():
            if edge in visited_edges:
                continue
            points, is_closed = self._walk_remaining_edge(
                skeleton_graph, edge, visited_edges
            )
            raw_paths.append(
                self._make_path(
                    skeleton_graph, len(raw_paths), points, is_closed=is_closed
                )
            )

        raw_paths.sort(key=self._path_sort_key)
        for path_id, path in enumerate(raw_paths):
            path.path_id = path_id

        paths, merge_count = self._merge_short_terminal_spurs(
            skeleton_graph, raw_paths
        )
        self.metrics = {
            "raw_path_count": len(raw_paths),
            "path_count": len(paths),
            "closed_loop_count": sum(path.is_closed for path in paths),
            "short_path_count": sum(path.is_short for path in paths),
            "merged_path_count": merge_count,
            "merged_source_path_count": sum(
                len(path.merged_from_path_ids)
                for path in paths
                if path.merged_from_path_ids
            ),
            "attached_loop_count": sum(
                path.is_closed and path.start_node_type == "attached_loop"
                for path in paths
            ),
        }
        return paths


class TraceLandmarkExtractor:
    """
    Extract ordered geometric landmarks from TracePath objects.

    Output includes:
    - start/end
    - global boundaries
    - local vertical extrema: top peaks and bottom valleys
    - local horizontal extrema: left/right turns
    """

    def __init__(self, settings=None):
        self.settings = normalize_trace_settings(settings)

    def _best_index(self, points, key_fn):
        best_index = 0
        best_key = key_fn(points[0], 0)

        for index, point in enumerate(points):
            current_key = key_fn(point, index)

            if current_key < best_key:
                best_key = current_key
                best_index = index

        return best_index

    def _make_landmark(
        self,
        landmark_id,
        path,
        landmark_type,
        index_on_path,
        prominence=0,
    ):
        return TraceLandmark(
            landmark_id=landmark_id,
            path_id=path.path_id,
            point=path.points[index_on_path],
            landmark_type=landmark_type,
            index_on_path=index_on_path,
            prominence=prominence,
        )

    def _global_landmark_specs(self, path):
        points = path.points

        # Secondary coordinates and path indices break ties consistently when
        # several pixels share the same outermost x or y coordinate.
        return [
            ("start", 0, 0),
            ("end", len(points) - 1, 0),
            (
                "global_left",
                self._best_index(
                    points,
                    lambda point, index: (point.x, point.y, index),
                ),
                0,
            ),
            (
                "global_right",
                self._best_index(
                    points,
                    lambda point, index: (-point.x, point.y, index),
                ),
                0,
            ),
            (
                "global_top",
                self._best_index(
                    points,
                    lambda point, index: (point.y, point.x, index),
                ),
                0,
            ),
            (
                "global_bottom",
                self._best_index(
                    points,
                    lambda point, index: (-point.y, point.x, index),
                ),
                0,
            ),
        ]

    def _local_landmark_specs(self, path):
        signal = PathSignal(path)

        specs = []

        y_extrema = signal.local_extrema_indices(
            axis="y",
            min_prominence=self.settings.local_extrema_min_prominence,
            min_spacing=self.settings.local_extrema_min_spacing,
        )

        for extrema in y_extrema:
            if extrema["kind"] == "min":
                landmark_type = "local_top_peak"
            else:
                landmark_type = "local_bottom_valley"

            specs.append(
                (
                    landmark_type,
                    extrema["index"],
                    extrema["prominence"],
                )
            )

        x_extrema = signal.local_extrema_indices(
            axis="x",
            min_prominence=self.settings.local_extrema_min_prominence,
            min_spacing=self.settings.local_extrema_min_spacing,
        )

        for extrema in x_extrema:
            if extrema["kind"] == "min":
                landmark_type = "local_left_turn"
            else:
                landmark_type = "local_right_turn"

            specs.append(
                (
                    landmark_type,
                    extrema["index"],
                    extrema["prominence"],
                )
            )

        specs.sort(key=lambda item: (item[1], item[0]))
        return specs

    def extract_for_path(self, path, starting_landmark_id):
        landmarks = []
        next_id = starting_landmark_id

        specs = []
        specs.extend(self._global_landmark_specs(path))
        specs.extend(self._local_landmark_specs(path))

        specs.sort(
            key=lambda item: (
                item[1],
                item[0],
            )
        )

        seen = set()

        for landmark_type, index_on_path, prominence in specs:
            key = (landmark_type, index_on_path)

            if key in seen:
                continue

            seen.add(key)

            landmarks.append(
                self._make_landmark(
                    landmark_id=next_id,
                    path=path,
                    landmark_type=landmark_type,
                    index_on_path=index_on_path,
                    prominence=prominence,
                )
            )

            next_id += 1

        return landmarks

    def extract_landmarks(self, trace_paths):
        landmarks = []
        next_id = 0

        for path in sorted(trace_paths, key=lambda item: item.path_id):
            path_landmarks = self.extract_for_path(
                path=path,
                starting_landmark_id=next_id,
            )

            landmarks.extend(path_landmarks)
            next_id += len(path_landmarks)

        return landmarks


