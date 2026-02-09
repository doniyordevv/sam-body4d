# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import cv2
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from utils.painter import color_list

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)


def visualize_sample(img_cv2, outputs, faces, id_current):
	img_mesh = img_cv2.copy()

	if outputs is None:
		return img_mesh

	rend_img = []
	for pid, person_output in enumerate(outputs):
		renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
		img2 = (
			renderer(
				person_output["pred_vertices"],
				person_output["pred_cam_t"],
				img_mesh.copy(),
				mesh_base_color=color_list[id_current[pid]+4],
				scene_bg_color=(0, 0, 0),
			)
			* 255
		)

		# Draw skeleton on top of rendered mesh
		img2 = _draw_skeletons(img2, [person_output])

		# cur_img = np.concatenate([img_cv2, img1, img2, img3], axis=1)
		rend_img.append(img2)

	return rend_img

def _draw_skeletons(image, outputs):
	"""Draw 2D skeleton overlay for all persons onto the image."""
	for person_output in outputs:
		kpts_2d = person_output.get("pred_keypoints_2d")
		if kpts_2d is None:
			continue
		# draw_skeleton expects (B, N, 3) where last col is confidence
		conf = np.ones((kpts_2d.shape[0], 1), dtype=kpts_2d.dtype)
		kpts_with_conf = np.concatenate([kpts_2d, conf], axis=-1)[None]  # (1, 70, 3)
		image = visualizer.draw_skeleton(image, kpts_with_conf, kpt_thr=0.0)
	return image

def visualize_sample_together(img_cv2, outputs, faces, id_current):
	# Render everything together
	img_mesh = img_cv2.copy()

	if outputs is None:
		return img_mesh

	# First, sort by depth, furthest to closest
	try:
		all_depths = np.stack([tmp['pred_cam_t'] for tmp in outputs], axis=0)[:, 2]
	except:
		return img_mesh
	outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

	id_sorted = np.argsort(-all_depths)   # by id not depth for consistent coloring

	# Then, put all meshes together as one super mesh
	all_pred_vertices = []
	all_faces = []
	all_color = []
	for pid, person_output in enumerate(outputs_sorted):
		all_pred_vertices.append(person_output["pred_vertices"] + person_output["pred_cam_t"])
		all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
		all_color.append(color_list[id_current[id_sorted[pid]]+4])
	all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
	all_faces = np.concatenate(all_faces, axis=0)

	# Pull out a fake translation; take the closest two
	fake_pred_cam_t = (np.max(all_pred_vertices[-2*18439:], axis=0) + np.min(all_pred_vertices[-2*18439:], axis=0)) / 2
	all_pred_vertices = all_pred_vertices - fake_pred_cam_t
	
	# Render front view
	renderer = Renderer(focal_length=person_output["focal_length"], faces=all_faces)
	img_mesh = (
		renderer(
			all_pred_vertices,
			fake_pred_cam_t,
			img_mesh,
			# mesh_base_color=LIGHT_BLUE,
			mesh_base_color=all_color,
			scene_bg_color=(0, 0, 0),
		)
		* 255
	)

	# Draw skeletons on top of rendered meshes
	img_mesh = _draw_skeletons(img_mesh, outputs)

	return img_mesh
