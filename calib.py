import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os
import yaml
import json
import argparse
from datetime import datetime

class ArUco3DProcessor:
    def __init__(self, intrinsic, data_dir, output_dir, debug_mode=False):
        self.intrinsic = intrinsic
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.left_dir = os.path.join(data_dir, "left")
        self.right_dir = os.path.join(data_dir, "right")
        self.ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.ARUCO_PARAMS = cv2.aruco.DetectorParameters()
        self.debug_mode = debug_mode

    def debug(self, message):
        if self.debug_mode:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[DEBUG] {timestamp} - {message}")

    def detect_aruco(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_corners, aruco_ids, _ = cv2.aruco.detectMarkers(
            gray, self.ARUCO_DICT, parameters=self.ARUCO_PARAMS
        )
        self.debug(f"Detected {len(aruco_corners)} ArUco markers.")
        return aruco_corners, aruco_ids

    def project_to_2d(self, pcd):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        projected_points = []
        projected_colors = []

        for point, color in zip(points, colors):
            x, y, z = point
            if z <= 0:
                continue
            u = self.intrinsic[0, 0] * x / z + self.intrinsic[0, 2]
            v = self.intrinsic[1, 1] * y / z + self.intrinsic[1, 2]
            projected_points.append([u, v])
            projected_colors.append(color)

        self.debug(f"Projected {len(projected_points)} points to 2D.")
        return np.array(projected_points, dtype=np.float32), np.array(projected_colors, dtype=np.float32)

    def calculate_rmse(self, points1, points2):
        rmse = np.sqrt(np.mean(np.linalg.norm(points1 - points2, axis=1) ** 2))
        self.debug(f"Calculated RMSE: {rmse}")
        return rmse

    def estimate_transformation(self, matched_pairs):
        left_points = np.array([pair[0] for pair in matched_pairs])
        right_points = np.array([pair[1] for pair in matched_pairs])

        centroid_left = np.mean(left_points, axis=0)
        centroid_right = np.mean(right_points, axis=0)

        H = ((right_points - centroid_right).T @ (left_points - centroid_left))

        U, S, Vt = np.linalg.svd(H)
        R_matrix = Vt.T @ U.T

        if np.linalg.det(R_matrix) < 0:
            Vt[-1, :] *= -1
            R_matrix = Vt.T @ U.T

        t_vector = centroid_left - R_matrix @ centroid_right

        transformation_matrix = np.identity(4)
        transformation_matrix[:3, :3] = R_matrix
        transformation_matrix[:3, 3] = t_vector

        return transformation_matrix

    def check_flatness(self, pcd, threshold=0.01):
        points = np.asarray(pcd.points)
        if len(points) < 3:
            raise ValueError("Not enough points to determine flatness.")

        # Compute the plane using SVD
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        _, _, vh = np.linalg.svd(centered_points, full_matrices=False)
        normal = vh[-1, :]

        # Calculate distances from the plane
        distances = np.dot(centered_points, normal)

        # Check if all points are within the threshold distance from the plane
        flat = np.all(np.abs(distances) < threshold)

        self.debug(f"Flatness check result: {flat}")

        # Add normal as an arrow in the point cloud for visualization
        arrow_length = 0.1
        normal_end = centroid + normal * arrow_length

        normal_pcd = o3d.geometry.PointCloud()
        normal_pcd.points = o3d.utility.Vector3dVector([centroid, normal_end])
        normal_pcd.paint_uniform_color([1, 0, 0])  # Red for normal vector

        merged_pcd = pcd + normal_pcd

        return flat, merged_pcd

    def process(self):
        # Load all files from directories
        left_files = sorted([f for f in os.listdir(self.left_dir) if f.endswith('.ply')])
        right_files = sorted([f for f in os.listdir(self.right_dir) if f.endswith('.ply')])

        if len(left_files) != len(right_files):
            raise ValueError("The number of left and right files must match.")

        self.debug(f"Found {len(left_files)} files in both left and right directories.")

        results = []

        for left_file, right_file in zip(left_files, right_files):
            self.debug(f"Processing {left_file} and {right_file}.")
            left_pcd = o3d.io.read_point_cloud(os.path.join(self.left_dir, left_file))
            right_pcd = o3d.io.read_point_cloud(os.path.join(self.right_dir, right_file))

            # Save point clouds for visualization
            o3d.io.write_point_cloud(os.path.join(self.output_dir, f"{left_file.split('.')[0]}.ply"), left_pcd)
            o3d.io.write_point_cloud(os.path.join(self.output_dir, f"{right_file.split('.')[0]}.ply"), right_pcd)

            # Process the rest of the pipeline (projection, matching, etc.)
            image1_projected, colors1 = self.project_to_2d(left_pcd)
            image2_projected, colors2 = self.project_to_2d(right_pcd)

            image1 = self.points_to_image(image1_projected, colors1, 1280, 720)
            image2 = self.points_to_image(image2_projected, colors2, 1280, 720)

            # ArUco marker detection
            aruco_corners1, aruco_ids1 = self.detect_aruco(image1)
            aruco_corners2, aruco_ids2 = self.detect_aruco(image2)

            detected_image1 = image1.copy()
            detected_image2 = image2.copy()
            if aruco_ids1 is not None:
                detected_image1 = cv2.aruco.drawDetectedMarkers(detected_image1, aruco_corners1, aruco_ids1)
            if aruco_ids2 is not None:
                detected_image2 = cv2.aruco.drawDetectedMarkers(detected_image2, aruco_corners2, aruco_ids2)

            # Save detected images for debugging
            cv2.imwrite(os.path.join(self.output_dir, f"detected_{left_file.split('.')[0]}.png"), detected_image1)
            cv2.imwrite(os.path.join(self.output_dir, f"detected_{right_file.split('.')[0]}.png"), detected_image2)

            if aruco_ids1 is not None and aruco_ids2 is not None:
                matched_ids = set(map(int, aruco_ids1.flatten())).intersection(map(int, aruco_ids2.flatten()))
                self.debug(f"Matched marker IDs: {matched_ids} (Count: {len(matched_ids)})")

                if not matched_ids:
                    self.debug("No matched marker IDs found. Skipping to next pair.")
                    continue

                matched_pairs = []

                for marker_id in matched_ids:
                    idx1 = np.where(aruco_ids1.flatten() == marker_id)[0][0]
                    idx2 = np.where(aruco_ids2.flatten() == marker_id)[0][0]
                    center1_2d = np.mean(aruco_corners1[idx1].reshape(-1, 2), axis=0)
                    center2_2d = np.mean(aruco_corners2[idx2].reshape(-1, 2), axis=0)

                    left_dists = np.linalg.norm(image1_projected - center1_2d, axis=1)
                    right_dists = np.linalg.norm(image2_projected - center2_2d, axis=1)
                    left_point = np.asarray(left_pcd.points)[np.argmin(left_dists)]
                    right_point = np.asarray(right_pcd.points)[np.argmin(right_dists)]

                    matched_pairs.append((left_point, right_point))

                if not matched_pairs:
                    self.debug("No matched pairs found after processing marker IDs. Skipping transformation.")
                    continue

                transformation_matrix = self.estimate_transformation(matched_pairs)
                self.debug(f"Estimated transformation matrix:\n{transformation_matrix}")

                # Apply transformation to right point cloud and merge
                #right_pcd.transform(transformation_matrix)
                #merged_pcd = left_pcd + right_pcd

                results.append({
                    'left_file': left_file,
                    'right_file': right_file,
                    'matched_ids': list(matched_ids),
                    'matched_pairs': [
                        [pair[0].tolist(), pair[1].tolist()] for pair in matched_pairs
                    ],
                    'transformation_matrix': transformation_matrix.tolist()
                })

                # Save intermediate results
                #o3d.io.write_point_cloud(os.path.join(self.output_dir, f"merged_{left_file.split('.')[0]}.ply"), merged_pcd)

        return results

    def points_to_image(self, projected_points, projected_colors, width, height):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for (u, v), color in zip(projected_points, projected_colors):
            if 0 <= int(v) < height and 0 <= int(u) < width:
                image[int(v), int(u)] = (color * 255).astype(np.uint8)
        self.debug(f"Generated 2D image with projected points.")
        return image


def calculate_final_transformation(results):
    all_left_points = []
    all_right_points = []

    for result in results:
        for left_point, right_point in result['matched_pairs']:
            all_left_points.append(left_point)
            all_right_points.append(right_point)

    all_left_points = np.array(all_left_points)
    all_right_points = np.array(all_right_points)

    # Estimate transformation using all matched pairs
    centroid_left = np.mean(all_left_points, axis=0)
    centroid_right = np.mean(all_right_points, axis=0)

    H = ((all_right_points - centroid_right).T @ (all_left_points - centroid_left))

    U, S, Vt = np.linalg.svd(H)
    R_matrix = Vt.T @ U.T

    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = Vt.T @ U.T

    t_vector = centroid_left - R_matrix @ centroid_right

    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = R_matrix
    transformation_matrix[:3, 3] = t_vector



    return transformation_matrix


def numpy_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [numpy_to_list(item) for item in data]
    elif isinstance(data, dict):
        return {key: numpy_to_list(value) for key, value in data.items()}
    elif isinstance(data, np.generic):
        return data.item()
    return data


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process ArUco marker 3D transformations.")
    parser.add_argument(
        "--data_directory",
        type=str,
        default=os.path.join(os.getcwd(), "data"),
        help="Path to the data directory containing input files. Default is './data'."
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default=os.path.join(os.getcwd(), "output"),
        help="Path to the output directory for saving results. Default is './output'."
    )
    args = parser.parse_args()

    # Use parsed arguments
    data_directory = args.data_directory
    output_directory = args.output_directory

    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Intrinsic matrix for camera
    intrinsic_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)

    # Initialize processor
    processor = ArUco3DProcessor(intrinsic_matrix, data_directory, output_directory, debug_mode=True)
    match_results = processor.process()

    # Print results
    for result in match_results:
        print(f"Processing {result['left_file']} and {result['right_file']}")
        print(f"Number of matched pairs: {len(result['matched_pairs'])}")
        print("Transformation Matrix:")
        print(result['transformation_matrix'])

    # Save match results to JSON
    match_results_cleaned = [numpy_to_list(entry) for entry in match_results]
    with open(os.path.join(output_directory, "match_results.json"), "w") as json_file:
        json.dump(match_results_cleaned, json_file)

    # Calculate final transformation across all data
    final_transformation = calculate_final_transformation(match_results)

    # Print final transformation
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DEBUG] {timestamp} - Final transformation matrix:\n{final_transformation}")
