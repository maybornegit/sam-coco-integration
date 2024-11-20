import os, sys, csv
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("/home/frc-ag-2/Downloads/sam2"))

from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from datetime import datetime
from sam2.build_sam import build_sam2_video_predictor
from matplotlib.patches import Rectangle
import time


class Tester():
	def __init__(self):
		# Initialize the node
		super().__init__()

		torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
		self.coredir = "/home/frc-ag-2/Downloads/sam2"



		self.model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
		self.model.eval()  # Set to evaluation mode

		self.device = torch.device("cuda")

		self.sam2_checkpoint = self.coredir + "/checkpoints/sam2.1_hiera_small.pt"
		self.model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
		self.predictor = None
		self.inference_state = None
		self.video_dir = self.coredir + "/notebooks/videos/bedroomtest"
		if not os.path.exists(self.video_dir):
			os.mkdir(self.video_dir)

		self.frame_dir = self.coredir + "/notebooks/videos/bedroom"
		self.update_csv = self.coredir+"/communication.csv"
		with open(self.update_csv, 'w') as file:
			writer = csv.writer(file)
			writer.writerow([300,300,0])

		with open(self.coredir + "/coco-labels-paper.txt", 'r') as file:
			# Read all lines and strip newlines, then store each line as an element in a list
			lines = file.readlines()

		# Strip newline characters and store each line as a string in a list
		self.class_names = [line.strip() for line in lines]

		self.frame_stride = 5
		self.init = -1

	def update_csv_file(self):
		### While loop checking until csv has 3 inputs
		update_ready = False
		startTime = time.time()
		with torch.no_grad():
			while True:
				with open(self.update_csv, 'r') as file:
					csv_reader = csv.reader(file)
					status = list(csv_reader)[0]
					if len(status) > 1:
						update_ready = True

				if update_ready:
					frame_idx = int(status[-1])
					frame = self.get_frame(frame_idx)
					if (frame_idx - self.init) >= self.frame_stride or self.init == -1:
						mask_combined = self.initialize_from_frame(frame)
						self.init = frame_idx
						print(time.time() - startTime)
						startTime = time.time()
					else:
						mask_combined = self.mask_step(frame, frame_idx-self.init)

					mask_combined /= 255
					mask_combined = mask_combined[:, :, 0]

					result = []
					for i in range((len(status)-1)//2):
						result.append(mask_combined[int(status[2*i]), int(status[2*i+1])])
					with open(self.update_csv, 'w') as file:
						writer = csv.writer(file)
						writer.writerow(result)
					print(result)

					with open(self.update_csv, 'w') as file:
						writer = csv.writer(file)
						status[-1] = int(status[-1]) + 1
						writer.writerow(status)
					update_ready = False

	def get_frame(self, frame_idx):
		"""
        needs to be re-written to allow for pulling from a video
        """
		filename = self.frame_dir +"/"+ f"{frame_idx:05d}.jpg"
		frame = Image.open(filename).convert("RGB")
		return frame

	def get_dynamic_segmentation(self, image):
		masks = []
		# Run inference on the image
		with torch.no_grad():
			prediction = self.model(image)

		# Process the results
		result = prediction[0]  # Get the first (and only) image result
		for i, class_id in enumerate(result['labels']):
			class_name = self.class_names[class_id.item()]

			# Check if the class is in the list of dynamic objects we care about
			dynamic_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
							   'boat']
			if class_name in dynamic_classes and result["scores"][i] > .9:
				object_mask = result['masks'][i, 0].cpu().numpy()
				object_mask = np.round(object_mask).astype(np.int32)
				masks.append(object_mask)

		return masks, result

	def prepare_data(
			self,
			img,
			image_size=1024,
			img_mean=(0.485, 0.456, 0.406),
			img_std=(0.229, 0.224, 0.225),
	):
		if isinstance(img, np.ndarray):
			img_np = img
			img_np = cv2.resize(img_np, (image_size, image_size)) / 255.0
			height, width = img.shape[:2]
		else:
			img_np = (
					np.array(img.convert("RGB").resize((image_size, image_size))) / 255.0
			)
			width, height = img.size
		img = torch.from_numpy(img_np).permute(2, 0, 1).float()

		img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
		img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
		img -= img_mean
		img /= img_std
		return img, width, height

	def coco_mask(self, im):
		# Load the input image
		# im = Image.open(image_path).convert("RGB")
		im = Image.fromarray(np.uint8(im))
		# Define the transformations for the input
		transform = transforms.Compose([transforms.ToTensor()])
		im_tensor = transform(im).unsqueeze(0)  # Add batch dimension

		masks_, result = self.get_dynamic_segmentation(im_tensor)
		return masks_

	def _get_feature(self, img, batch_size):
		if self.device == torch.device("cuda"):
			image = img.cuda().float().unsqueeze(0)
		else:
			image = img.cpu().float().unsqueeze(0)
		backbone_out = self.predictor.forward_image(image)
		expanded_image = image.expand(batch_size, -1, -1, -1)
		expanded_backbone_out = {
			"backbone_fpn": backbone_out["backbone_fpn"].copy(),
			"vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
		}
		for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
			expanded_backbone_out["backbone_fpn"][i] = feat.expand(
				batch_size, -1, -1, -1
			)
		for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
			pos = pos.expand(batch_size, -1, -1, -1)
			expanded_backbone_out["vision_pos_enc"][i] = pos

		features = self.predictor._prepare_backbone_features(expanded_backbone_out)
		features = (expanded_image,) + features
		return features

	def _get_orig_video_res_output(self, any_res_masks):
		"""
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
		device = self.inference_state["device"]
		video_H = self.inference_state["video_height"]
		video_W = self.inference_state["video_width"]
		any_res_masks = any_res_masks.to(device, non_blocking=True)
		if any_res_masks.shape[-2:] == (video_H, video_W):
			video_res_masks = any_res_masks
		else:
			video_res_masks = torch.nn.functional.interpolate(
				any_res_masks,
				size=(video_H, video_W),
				mode="bilinear",
				align_corners=False,
			)
		return any_res_masks, video_res_masks

	def initialize_from_frame(self, frame):
		# Load the pre-trained Mask R-CNN model
		self.predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint, device=self.device)

		## Create Subvideo
		frame.save(self.video_dir + "/00000.jpg")

		masks = self.coco_mask(frame)
		self.inference_state = self.predictor.init_state(video_path=self.video_dir)
		ann_frame_idx = 0  # the frame index we interact with
		ann_obj_id = 0  # give a unique id to each object we interact with (it can be any integers)
		# Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
		for i, m in enumerate(masks):
			mask = np.array(m, dtype=np.float32)
			_, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
				inference_state=self.inference_state,
				frame_idx=ann_frame_idx,
				obj_id=ann_obj_id,
				mask=mask,
			)
			ann_obj_id += 1

		video_segments = {}  # video_segments contains the per-frame segmentation results
		for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
			video_segments[out_frame_idx] = {
				out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
				for i, out_obj_id in enumerate(out_obj_ids)
			}

		mask_combined = np.zeros(np.array(frame).shape)  # Copy original image
		for mask in masks:
			# m = (mask > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.bool_)
			# if m.ndim == 3:
			# 	m = m[:, :, 0]  # Drop the third dimension if the mask is (height, width, 1)
			mask_combined[mask == 1] = [255, 0, 0]  # Add the red mask on top
		return mask_combined

	def mask_step(self, frame, i):
		im = Image.fromarray(np.uint8(frame)).convert("RGB")
		img, _, _ = self.prepare_data(im)
		output_dict = self.inference_state["output_dict"]
		batch_size = len(self.inference_state["obj_idx_to_id"])
		# # Retrieve correct image features
		(
			_,
			_,
			current_vision_feats,
			current_vision_pos_embeds,
			feat_sizes,
		) = self._get_feature(img, batch_size)

		current_out = self.predictor.track_step(
			frame_idx=i,
			is_init_cond_frame=False,
			current_vision_feats=current_vision_feats,
			current_vision_pos_embeds=current_vision_pos_embeds,
			feat_sizes=feat_sizes,
			point_inputs=None,
			mask_inputs=None,
			output_dict=output_dict,
			num_frames=self.inference_state["num_frames"] + i,
			track_in_reverse=False,
			run_mem_encoder=True,
			prev_sam_mask_logits=None,
		)
		pred_masks_gpu = current_out["pred_masks"]
		_, video_res_masks = self._get_orig_video_res_output(pred_masks_gpu)

		mask_combined = np.zeros(np.array(im).shape)  # Copy original image
		for mask in video_res_masks:
			m = (mask > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.bool_)
			if m.ndim == 3:
				m = m[:, :, 0]  # Drop the third dimension if the mask is (height, width, 1)
			mask_combined[m == 1] = [255, 0, 0]  # Add the red mask on top
		return mask_combined

# Non-class code
# def get_dynamic_segmentation(image):
# 	"""Generate a binary mask for the dynamic objects in the input image."""
# 	masks = []
#
# 	with open(coredir+"/coco-labels-paper.txt", 'r') as file:
# 		# Read all lines and strip newlines, then store each line as an element in a list
# 		lines = file.readlines()
#
# 	# Strip newline characters and store each line as a string in a list
# 	class_names = [line.strip() for line in lines]
#
# 	# Run inference on the image
# 	with torch.no_grad():
# 		prediction = model(image)
#
# 	# Process the results
# 	result = prediction[0]  # Get the first (and only) image result
# 	for i, class_id in enumerate(result['labels']):
# 		class_name = class_names[class_id.item()]
#
#
# 		# Check if the class is in the list of dynamic objects we care about
# 		dynamic_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']
# 		if class_name in dynamic_classes and result["scores"][i] > .9:
# 			object_mask = result['masks'][i, 0].cpu().numpy()
# 			object_mask = np.round(object_mask).astype(np.int32)
# 			masks.append(object_mask)
#
# 	return masks, result
#
# def prepare_data(
#         img,
#         image_size=1024,
#         img_mean=(0.485, 0.456, 0.406),
#         img_std=(0.229, 0.224, 0.225),
#     ):
#         if isinstance(img, np.ndarray):
#             img_np = img
#             img_np = cv2.resize(img_np, (image_size, image_size)) / 255.0
#             height, width = img.shape[:2]
#         else:
#             img_np = (
#                 np.array(img.convert("RGB").resize((image_size, image_size))) / 255.0
#             )
#             width, height = img.size
#         img = torch.from_numpy(img_np).permute(2, 0, 1).float()
#
#         img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
#         img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
#         img -= img_mean
#         img /= img_std
#         return img, width, height
#
# def coco_mask(im):
# 	# Load the input image
# 	# im = Image.open(image_path).convert("RGB")
# 	im = Image.fromarray(np.uint8(im))
# 	# Define the transformations for the input
# 	transform = transforms.Compose([transforms.ToTensor()])
# 	im_tensor = transform(im).unsqueeze(0)  # Add batch dimension
#
# 	masks_, result = get_dynamic_segmentation(im_tensor)
# 	return masks_
#
#
# def _get_feature(img, batch_size):
# 	image = img.cuda().float().unsqueeze(0)
# 	backbone_out = predictor.forward_image(image)
# 	expanded_image = image.expand(batch_size, -1, -1, -1)
# 	expanded_backbone_out = {
# 		"backbone_fpn": backbone_out["backbone_fpn"].copy(),
# 		"vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
# 	}
# 	for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
# 		expanded_backbone_out["backbone_fpn"][i] = feat.expand(
# 			batch_size, -1, -1, -1
# 		)
# 	for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
# 		pos = pos.expand(batch_size, -1, -1, -1)
# 		expanded_backbone_out["vision_pos_enc"][i] = pos
#
# 	features = predictor._prepare_backbone_features(expanded_backbone_out)
# 	features = (expanded_image,) + features
# 	return features
#
# def _get_orig_video_res_output(any_res_masks):
# 	"""
# 	Resize the object scores to the original video resolution (video_res_masks)
# 	and apply non-overlapping constraints for final output.
# 	"""
# 	device = inference_state["device"]
# 	video_H = inference_state["video_height"]
# 	video_W = inference_state["video_width"]
# 	print(video_H, video_W)
# 	any_res_masks = any_res_masks.to(device, non_blocking=True)
# 	if any_res_masks.shape[-2:] == (video_H, video_W):
# 		video_res_masks = any_res_masks
# 	else:
# 		video_res_masks = torch.nn.functional.interpolate(
# 			any_res_masks,
# 			size=(video_H, video_W),
# 			mode="bilinear",
# 			align_corners=False,
# 		)
# 	return any_res_masks, video_res_masks
#
# def initialize_from_frame(frame, device, sam2_checkpoint, model_cfg, coredir):
# 	# Load the pre-trained Mask R-CNN model
# 	predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
#
# 	## Create Subvideo
# 	video_dir = coredir+"/notebooks/videos/bedroomtest"
# 	frame.save(video_dir+"/00000.jpg")
#
# 	masks = coco_mask(frame)
# 	inference_state = predictor.init_state(video_path=video_dir)
# 	ann_frame_idx = 0  # the frame index we interact with
# 	ann_obj_id = 0  # give a unique id to each object we interact with (it can be any integers)
# 	# Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
# 	for i, m in enumerate(masks):
# 		mask = np.array(m, dtype=np.float32)
# 		_, out_obj_ids, out_mask_logits = predictor.add_new_mask(
# 			inference_state=inference_state,
# 			frame_idx=ann_frame_idx,
# 			obj_id=ann_obj_id,
# 			mask=mask,
# 		)
# 		ann_obj_id += 1
#
# 	video_segments = {}  # video_segments contains the per-frame segmentation results
# 	for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
# 		video_segments[out_frame_idx] = {
# 			out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
# 			for i, out_obj_id in enumerate(out_obj_ids)
# 		}
# 	return masks, inference_state, predictor
#
# def mask_step(frame, inference_state, predictor):
# 	im = Image.fromarray(np.uint8(frame)).convert("RGB")
# 	img, _, _ = prepare_data(im)
# 	output_dict = inference_state["output_dict"]
# 	batch_size = len(inference_state["obj_idx_to_id"])
#
# 	# # Retrieve correct image features
# 	(
# 		_,
# 		_,
# 		current_vision_feats,
# 		current_vision_pos_embeds,
# 		feat_sizes,
# 	) = _get_feature(img, batch_size)
#
# 	current_out = predictor.track_step(
# 		frame_idx=i,
# 		is_init_cond_frame=False,
# 		current_vision_feats=current_vision_feats,
# 		current_vision_pos_embeds=current_vision_pos_embeds,
# 		feat_sizes=feat_sizes,
# 		point_inputs=None,
# 		mask_inputs=None,
# 		output_dict=output_dict,
# 		num_frames=inference_state["num_frames"] + i,
# 		track_in_reverse=False,
# 		run_mem_encoder=True,
# 		prev_sam_mask_logits=None,
# 	)
# 	pred_masks_gpu = current_out["pred_masks"]
# 	_, video_res_masks = _get_orig_video_res_output(pred_masks_gpu)
#
# 	mask_combined = np.zeros(np.array(im).shape)  # Copy original image
# 	for mask in video_res_masks:
# 		m = (mask > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.bool_)
# 		if m.ndim == 3:
# 			m = m[:, :, 0]  # Drop the third dimension if the mask is (height, width, 1)
# 		print(mask_combined.shape, m.shape)
# 		mask_combined[m == 1] = [255, 0, 0]  # Add the red mask on top
# 	return mask_combined

# # Function to extract frames from video (redundant with matrix loading)
# def extract_frames(video_path, output_dir, frame_range):
# 	# Create the output directory if it doesn't exist
# 	if not os.path.exists(output_dir):
# 		os.makedirs(output_dir)
#
# 	# Open the video file
# 	video_capture = cv2.VideoCapture(video_path)
#
# 	# Check if the video opened successfully
# 	if not video_capture.isOpened():
# 		print(f"Error: Could not open video file {video_path}")
# 		return
#
# 	frame_count = 0
# 	while True:
# 		# Read a frame from the video
# 		ret, frame = video_capture.read()
#
# 		# If the frame was read successfully, ret will be True
# 		if not ret or (frame_count > max(frame_range) and frame_range != []):
# 			break  # End of video
#
# 		# Save the frame as an image
# 		if frame_count in frame_range or frame_range == []:
# 			frame_filename = os.path.join(output_dir, f"{frame_count:05d}.jpg")
# 			cv2.imwrite(frame_filename, frame)
# 			print(f"Saved {frame_filename}")
#
# 		frame_count += 1
#
# 	# Release the video capture object
# 	video_capture.release()
# 	print(f"Extracted {frame_count} frames to {output_dir}")

if __name__ == '__main__':
	sam = Tester()
	sam.update_csv_file()
	# frame = Image.open(sam.coredir + "/notebooks/videos/bedroom/0000" + str(0) + ".jpg").convert("RGB")
	# mask_combined = sam.initialize_from_frame(frame)
	# print(mask_combined[300, 300])
	# with torch.no_grad():
	# 	for i in range(1,9):
	# 		frame = Image.open(sam.coredir+"/0000"+str(i)+".jpg").convert("RGB")
	# 		mask_combined = sam.mask_step(frame, i)
	# 		print(mask_combined[300,300])

			# img = Image.open(sam.coredir+"/0000"+str(i)+".jpg").convert("RGB")
			# im = np.clip(img, 0, 255) # Ensure pixel values are between 0 and 1
			# im = (im).astype(np.uint8)  # Convert to 8-bit unsigned integers (0-255)
			# mask_combined /= 255
			# mask_combined = mask_combined[:,:,0]
			#
			# combined_image = np.copy(im)  # Copy original image
			# combined_image[mask_combined == 1] = [255, 0, 0]  # Add the red mask on top
			#
			# # Display the combined image
			# fig, ax = plt.subplots()
			# plt.imshow(combined_image)
			# plt.title("Image with Mask Overlaid")
			# plt.axis('off')  # Hide axes
			# plt.show()


	# torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
	# coredir = "/home/frc-ag-2/Downloads/sam2"
	#
	# model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
	# model.eval()  # Set to evaluation mode
	#
	# device = torch.device("cuda")
	#
	# sam2_checkpoint = coredir + "/checkpoints/sam2.1_hiera_small.pt"
	# model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
	#
	# frame = Image.open(coredir + "/notebooks/videos/bedroom/0000" + str(0) + ".jpg").convert("RGB")
	# masks, inference_state, predictor = initialize_from_frame(frame, device, sam2_checkpoint, model_cfg, coredir)
	# with torch.no_grad():
	# 	for i in range(1,9):
	# 		frame = Image.open(coredir+"/0000"+str(i)+".jpg").convert("RGB")
	# 		mask_combined = mask_step(frame, inference_state, predictor)


	## Previous Code
	# with open(coredir+"/coco-labels-paper.txt", 'r') as file:
	# 	# Read all lines and strip newlines, then store each line as an element in a list
	# 	lines = file.readlines()
	#
	# # Strip newline characters and store each line as a string in a list
	# class_names = [line.strip() for line in lines]

	# Load the pre-trained Mask R-CNN model


	# print('Initializing Mask R-CNN network...')
	# startTime = time.time()
	# predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

	# ## Create Subvideo
	# video_path = coredir+"/notebooks/videos/bedroom.mp4"
	# video_dir = coredir+"/notebooks/videos/bedroomtest"
	# frame_range = [0]
	# extract_frames(video_path, video_dir, frame_range)  ### Frame Range will be from core file to file_of interest
	#
	# # scan all the JPEG frame names in this directory
	# frame_names = [
	# 	p for p in os.listdir(video_dir)
	# 	if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
	# ]
	# frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
	# print(time.time() - startTime, "seconds to complete - writing files")
	# startTime = time.time()
	#
	# masks = coco_mask(Image.open(os.path.join(video_dir, frame_names[0])))
	# print(time.time() - startTime, "seconds to complete - initial masks")
	# startTime = time.time()
	#
	# inference_state = predictor.init_state(video_path=video_dir)
	# print(time.time() - startTime, "seconds to complete - initalization")
	# startTime = time.time()
	#
	# ann_frame_idx = 0  # the frame index we interact with
	# ann_obj_id = 0  # give a unique id to each object we interact with (it can be any integers)
	# # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
	# for i, m in enumerate(masks):
	# 	mask = np.array(m, dtype=np.float32)
	# 	_, out_obj_ids, out_mask_logits = predictor.add_new_mask(
	# 		inference_state=inference_state,
	# 		frame_idx=ann_frame_idx,
	# 		obj_id=ann_obj_id,
	# 		mask=mask,
	# 	)
	# 	ann_obj_id += 1
	# print(time.time() - startTime, "seconds to complete - adding predictor")
	# startTime = time.time()

	# video_segments = {}  # video_segments contains the per-frame segmentation results
	# for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
	# 	video_segments[out_frame_idx] = {
	# 		out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
	# 		for i, out_obj_id in enumerate(out_obj_ids)
	# 	}
	# print(time.time() - startTime, "seconds to complete - propagate")
	# startTime = time.time()
	# with torch.no_grad():
	# 	for i in range(1,9):
	# 		img = Image.open(coredir+"/0000"+str(i)+".jpg").convert("RGB")
			# img, _, _ = prepare_data(img)
			# output_dict = inference_state["output_dict"]
			# consolidated_frame_inds = inference_state["consolidated_frame_inds"]
			# obj_ids = inference_state["obj_ids"]
			# num_frames = inference_state["num_frames"]
			# output_dict = inference_state["output_dict"]
			# obj_ids = inference_state["obj_ids"]
			# batch_size = len(inference_state["obj_idx_to_id"])
			#
			# # # Retrieve correct image features
			# (
			# 	_,
			# 	_,
			# 	current_vision_feats,
			# 	current_vision_pos_embeds,
			# 	feat_sizes,
			# ) = _get_feature(img, batch_size)
			#
			# current_out = predictor.track_step(
			# 	frame_idx=i,
			# 	is_init_cond_frame=False,
			# 	current_vision_feats=current_vision_feats,
			# 	current_vision_pos_embeds=current_vision_pos_embeds,
			# 	feat_sizes=feat_sizes,
			# 	point_inputs=None,
			# 	mask_inputs=None,
			# 	output_dict=output_dict,
			# 	num_frames=inference_state["num_frames"]+i,
			# 	track_in_reverse=False,
			# 	run_mem_encoder=True,
			# 	prev_sam_mask_logits=None,
			# )
			# print(len(inference_state["images"]))
			# storage_device = inference_state["storage_device"]
			# pred_masks_gpu = current_out["pred_masks"]
			# pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
			# _, video_res_masks = _get_orig_video_res_output(pred_masks_gpu)
			# print(time.time() - startTime, "seconds to complete - fastmask")
			# startTime = time.time()
			#
			# img = Image.open(coredir+"/0000"+str(i)+".jpg").convert("RGB")
			# im = np.clip(img, 0, 255) # Ensure pixel values are between 0 and 1
			# im = (im).astype(np.uint8)  # Convert to 8-bit unsigned integers (0-255)
			#
			# combined_image = np.copy(im)  # Copy original image
			# for mask in video_res_masks:
			# 	m = (mask>0.0).permute(1,2,0).cpu().numpy().astype(np.bool_)
			# 	if m.ndim == 3:
			# 		m = m[:, :, 0]  # Drop the third dimension if the mask is (height, width, 1)
			# 	combined_image[m == 1] = [255, 0, 0]  # Add the red mask on top
			#
			# # Display the combined image
			# fig, ax = plt.subplots()
			# plt.imshow(combined_image)
			# plt.title("Image with Mask Overlaid")
			# plt.axis('off')  # Hide axes
			# plt.show()


    

