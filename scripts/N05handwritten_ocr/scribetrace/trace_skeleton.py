"""Topology-preserving thinning and deterministic skeleton graph building."""

import numpy as np

from .trace_common import NEIGHBOR_OFFSETS, coordinate_key, edge_key
from .trace_models import SkeletonPoint
from .trace_settings import normalize_trace_settings

_coordinate_key = coordinate_key

class TraceSkeletonizer:
    """Thin binary ink with topology-preserving Zhang-Suen iterations."""

    def __init__(self, settings=None):
        self.settings = normalize_trace_settings(settings)

    def normalize_binary_mask(self, mask):
        return (np.asarray(mask) > 128).astype(np.uint8)

    @staticmethod
    def _transition_count(neighbors):
        return sum(
            neighbors[index] == 0 and neighbors[(index + 1) % 8] == 1
            for index in range(8)
        )

    def _subiteration(self, image, first_pass):
        removals = []
        height, width = image.shape

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if image[y, x] != 1:
                    continue
                p2 = image[y - 1, x]
                p3 = image[y - 1, x + 1]
                p4 = image[y, x + 1]
                p5 = image[y + 1, x + 1]
                p6 = image[y + 1, x]
                p7 = image[y + 1, x - 1]
                p8 = image[y, x - 1]
                p9 = image[y - 1, x - 1]
                neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
                neighbor_count = int(sum(neighbors))

                if not 2 <= neighbor_count <= 6:
                    continue
                if self._transition_count(neighbors) != 1:
                    continue

                if first_pass:
                    first_product = p2 * p4 * p6
                    second_product = p4 * p6 * p8
                else:
                    first_product = p2 * p4 * p8
                    second_product = p2 * p6 * p8

                if first_product == 0 and second_product == 0:
                    removals.append((y, x))

        for y, x in removals:
            image[y, x] = 0
        return bool(removals)

    def skeletonize(self, mask):
        """Return a one-pixel white centerline while preserving topology."""
        binary = self.normalize_binary_mask(mask)
        padded = np.pad(binary, 1, mode="constant")

        while True:
            changed_first = self._subiteration(padded, True)
            changed_second = self._subiteration(padded, False)
            if not changed_first and not changed_second:
                break

        return (padded[1:-1, 1:-1] * 255).astype(np.uint8)

    def skeletonize_from_mask_path(self, mask_path):
        adapter = TraceMaskAdapter(self.settings)
        return self.skeletonize(adapter.load_trace_mask(mask_path))


class SkeletonPointExtractor:
    """Convert white skeleton pixels into sorted SkeletonPoint objects."""

    def is_skeleton_pixel(self, pixel_value):
        return int(np.asarray(pixel_value).flat[0]) > 128

    def extract_points(self, skeleton_mask):
        coordinates = np.argwhere(np.asarray(skeleton_mask) > 128)
        return [SkeletonPoint(x, y) for y, x in coordinates]


class SkeletonGraph:
    """Build a deterministic pruned 8-neighbor skeleton graph."""

    def __init__(self, points):
        self.points = sorted(list(points or []), key=_coordinate_key)
        self.point_lookup = {
            point.to_tuple(): point
            for point in self.points
        }
        self.neighbor_map = self.build_neighbor_map()
        self._crossing_numbers = {
            point.to_tuple(): self._calculate_crossing_number(point)
            for point in self.points
        }
        self._roles = {
            point.to_tuple(): self._classify_point(point)
            for point in self.points
        }
        self._junction_clusters = self._build_junction_clusters()
        self._junction_cluster_lookup = {
            point.to_tuple(): cluster_id
            for cluster_id, cluster in enumerate(self._junction_clusters)
            for point in cluster
        }

    def _should_connect(self, coordinate_a, coordinate_b):
        dx = coordinate_b[0] - coordinate_a[0]
        dy = coordinate_b[1] - coordinate_a[1]
        if max(abs(dx), abs(dy)) != 1:
            return False
        if dx == 0 or dy == 0:
            return True

        # A diagonal is redundant when either orthogonal corner pixel already
        # provides a two-step bridge. Removing it prevents false junctions.
        bridge_a = (coordinate_a[0] + dx, coordinate_a[1])
        bridge_b = (coordinate_a[0], coordinate_a[1] + dy)
        return bridge_a not in self.point_lookup and bridge_b not in self.point_lookup

    def build_neighbor_map(self):
        neighbor_map = {}
        for point in self.points:
            coordinate = point.to_tuple()
            neighbors = []
            for dx, dy in NEIGHBOR_OFFSETS:
                candidate = (point.x + dx, point.y + dy)
                if (
                    candidate in self.point_lookup
                    and self._should_connect(coordinate, candidate)
                ):
                    neighbors.append(self.point_lookup[candidate])
            neighbor_map[coordinate] = sorted(neighbors, key=_coordinate_key)
        return neighbor_map

    def _calculate_crossing_number(self, point):
        """Count locally disconnected neighbor arcs after diagonal pruning."""
        neighbor_coordinates = {
            neighbor.to_tuple()
            for neighbor in self.neighbors_of(point)
        }
        if not neighbor_coordinates:
            return 0

        unvisited = set(neighbor_coordinates)
        arc_count = 0
        while unvisited:
            start = min(unvisited, key=lambda value: (value[1], value[0]))
            unvisited.remove(start)
            stack = [start]
            arc_count += 1
            while stack:
                current = stack.pop()
                for neighbor in self.neighbor_map.get(current, []):
                    coordinate = neighbor.to_tuple()
                    if coordinate in unvisited and coordinate != point.to_tuple():
                        unvisited.remove(coordinate)
                        stack.append(coordinate)
        return arc_count

    def _classify_point(self, point):
        degree = self.neighbor_count(point)
        crossing_number = self._crossing_numbers[point.to_tuple()]
        if degree == 0:
            return "isolated"
        if degree == 1:
            return "endpoint"
        if crossing_number >= 3:
            return "junction"
        return "path"

    def _build_junction_clusters(self):
        junction_lookup = {
            point.to_tuple(): point
            for point in self.points
            if self._roles[point.to_tuple()] == "junction"
        }
        unvisited = set(junction_lookup)
        clusters = []

        while unvisited:
            start = min(unvisited, key=lambda value: (value[1], value[0]))
            unvisited.remove(start)
            stack = [junction_lookup[start]]
            cluster = []
            while stack:
                current = stack.pop()
                cluster.append(current)
                for neighbor in reversed(self.neighbors_of(current)):
                    coordinate = neighbor.to_tuple()
                    if coordinate in unvisited and coordinate in junction_lookup:
                        unvisited.remove(coordinate)
                        stack.append(junction_lookup[coordinate])
            clusters.append(sorted(cluster, key=_coordinate_key))
        return clusters

    def neighbor_count(self, point):
        return len(self.neighbor_map.get(point.to_tuple(), []))

    def neighbors_of(self, point):
        return self.neighbor_map.get(point.to_tuple(), [])

    def crossing_number(self, point):
        return self._crossing_numbers.get(point.to_tuple(), 0)

    def endpoints(self):
        return [point for point in self.points if self.is_endpoint(point)]

    def junctions(self):
        return [point for point in self.points if self.is_junction_point(point)]

    def path_points(self):
        return [point for point in self.points if self.is_path_point(point)]

    def isolated_points(self):
        return [
            point
            for point in self.points
            if self._roles[point.to_tuple()] == "isolated"
        ]

    def junction_clusters(self):
        return self._junction_clusters

    def junction_cluster_lookup(self):
        return self._junction_cluster_lookup

    def junction_cluster_id_for_point(self, point):
        return self._junction_cluster_lookup.get(point.to_tuple())

    def junction_cluster_centers(self):
        centers = []
        for cluster in self._junction_clusters:
            centers.append(
                SkeletonPoint(
                    round(sum(point.x for point in cluster) / len(cluster)),
                    round(sum(point.y for point in cluster) / len(cluster)),
                )
            )
        return centers

    def is_junction_point(self, point):
        return self._roles.get(point.to_tuple()) == "junction"

    def is_endpoint(self, point):
        return self._roles.get(point.to_tuple()) == "endpoint"

    def is_path_point(self, point):
        return self._roles.get(point.to_tuple()) == "path"

    def all_edges(self):
        """Return every undirected graph edge in deterministic order."""
        edges = set()
        for point in self.points:
            for neighbor in self.neighbors_of(point):
                edges.add(edge_key(point, neighbor))
        return sorted(edges)

    def to_dict(self):
        crossing_histogram = {}
        for value in self._crossing_numbers.values():
            key = str(value)
            crossing_histogram[key] = crossing_histogram.get(key, 0) + 1
        return {
            "point_count": len(self.points),
            "edge_count": len(self.all_edges()),
            "endpoint_count": len(self.endpoints()),
            "junction_pixel_count": len(self.junctions()),
            "junction_cluster_count": len(self._junction_clusters),
            "path_point_count": len(self.path_points()),
            "isolated_point_count": len(self.isolated_points()),
            "crossing_number_histogram": crossing_histogram,
        }

    def to_debug_dict(self, include_points=False):
        data = self.to_dict()
        if include_points:
            data["endpoints"] = [point.to_dict() for point in self.endpoints()]
            data["junctions"] = [point.to_dict() for point in self.junctions()]
            data["isolated_points"] = [
                point.to_dict() for point in self.isolated_points()
            ]
        return data

