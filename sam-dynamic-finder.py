import os, sys, csv
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("/home/frc-ag-2/Downloads/sam2")) ### CHANGE DEPENDING ON LOCAL LOCATION

from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from datetime import datetime
from sam2.build_sam import build_sam2_video_predictor
from matplotlib.patches import Rectangle
import time


class SAMDynamicReader():
	def __init__(self):
		"""
		SAMFrameReader Functionalities:
		- get_frame is an early build, meant to call the relevant frame for segmentation
		- coco_mask uses a pre-trained Mask RCNN to find "dynamic-likely" objects and creates masks for SAM initialization
		- initialize_first_frame initializes the SAM2 Predictor and inputs the relevant object masks
		- mask_step takes the encoded masks and transfers them to a new frame
		- update_csv_file packages mask_step and init_first_frame together to continually read and write a csv with whether a pixel location is in a mask
		"""
		super().__init__()

		## initialize directories, device, and relevant class information
		# torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
		self.coredir = "/home/frc-ag-2/Downloads/sam2"   ### CHANGE DEPENDING ON LOCAL LOCATION

		self.model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
		self.model.eval()  # Set to evaluation mode

		self.device = torch.device("cpu")

		self.sam2_checkpoint = self.coredir + "/checkpoints/sam2.1_hiera_small.pt"
		self.model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
		self.predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint, device=self.device)
		self.inference_state = None
		self.video_dir = self.coredir + "/notebooks/videos/bedroomtest"   # where a single frame goes for future frame prediction
		if not os.path.exists(self.video_dir):
			os.mkdir(self.video_dir)

		### This block is structures for demo, definitely can be optimized
		self.frame_dir = self.coredir + "/notebooks/videos/bedroom"       # separated directory with all frames
		self.update_csv = self.coredir+"/communication.csv"               # communication csv
		with open(self.update_csv, 'w') as file:
			writer = csv.writer(file)
			writer.writerow([300,300,200,100,0])

		with open(self.coredir + "/coco-labels-paper.txt", 'r') as file:   ### REMEMBER TO COPY TXT INTO SAM2 DIR
			# Read all lines and strip newlines, then store each line as an element in a list
			lines = file.readlines()

		# Strip newline characters and store each line as a string in a list
		self.class_names = [line.strip() for line in lines]

		self.frame_stride = 5      ### increase for closer to real-time
		self.init = -1

	def update_csv_file(self):
		'''
		Purpose: Communication loop over csv file, writing whether a pixel loc. is in a dynamic object mask
		Application: Need another script writing to the csv. This scripts takes comma separated arguments and writes a boolean response
		Arguments: (In the csv) They should be organize x1, y1, x2, y2, ..., xn, yn, frame_idx
			- frame_idx is the number of the frame in the video seq.
		Returns: (In the csv) Sequence of booleans corresponding to each pair of pixel coor. provided

		Example: If 300,300,10,10,15 is found in the csv, then 1, 0 might be returned
		:return:
		'''
		### While loop checking until csv has 3 inputs
		update_ready = False
		old_result = []
		startTime = time.time()   ### timing variable
		with torch.no_grad():
			while True:
				## check whether anything new has been written
				with open(self.update_csv, 'r') as file:
					csv_reader = csv.reader(file)
					status = list(csv_reader)[0]
					if len(status) > 1 and status != [str(res) for res in old_result]:
						update_ready = True

				if update_ready:
					## choose whether to re-initialize or step
					frame_idx = int(status[-1])
					frame = self.get_frame(frame_idx)
					if (frame_idx - self.init) >= self.frame_stride or self.init == -1:
						mask_combined = self.initialize_from_frame(frame)
						self.init = frame_idx
						print(time.time() - startTime)  ### output cycle time
						startTime = time.time()
					else:
						mask_combined = self.mask_step(frame, frame_idx-self.init)

					mask_combined /= 255
					mask_combined = mask_combined[:, :, 0]

					## write mask results to csv
					result = []
					for i in range((len(status)-1)//2):
						result.append(mask_combined[int(status[2*i]), int(status[2*i+1])])
					with open(self.update_csv, 'w') as file:
						writer = csv.writer(file)
						writer.writerow(result)
					print(result)
					old_result = result[:]

					# Comment out if you want to stop continual "fake communication"
					with open(self.update_csv, 'w') as file:
						writer = csv.writer(file)
						status[-1] = int(status[-1]) + 1
						writer.writerow(status)
					update_ready = False

	def get_frame(self, frame_idx):
		"""
        Temporary ~ if possible, could be updated to read a mp4 file or soemething more real-time esque
        """
		filename = self.frame_dir +"/"+ f"{frame_idx:05d}.jpg"
		frame = Image.open(filename).convert("RGB")
		return frame

	def get_dynamic_segmentation(self, image):
		"""
		Core COCO Application ~ applies pre-trained model on input, finds which masks are high scoring and "dynamic", and adds them to a mask list
		"""
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
		"""
		For preparing a frame matrix to the mask_step pipeline
		"""
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
		"""
		Input an image and receive a list of masks that belong to high scoring dynamic objects
		"""
		# Load the input image
		# im = Image.open(image_path).convert("RGB")
		im = Image.fromarray(np.uint8(im))
		# Define the transformations for the input
		transform = transforms.Compose([transforms.ToTensor()])
		im_tensor = transform(im).unsqueeze(0)  # Add batch dimension

		masks_, result = self.get_dynamic_segmentation(im_tensor)
		return masks_

	def _get_feature(self, img, batch_size):
		"""
		Feature calling for mask_step function
		"""
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
		"""
		Core initialization every few frames for storing an image + mask inside the predictor's inference state
		Used for transfering this masks into the other frames
		Outputs a combined mask for the initializing frame ~ comes straight from the COCO model output
		"""
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
		"""
		With an initialized inference state in the class, the mask_step takes the encoded mask / image and takes "a step" to the next frame
		This takes in a frame (and a semi-arbitrary iteration #) and outputs a combined mask of dynamic objects
		"""
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

if __name__ == '__main__':
	### This is the core code for getting this node started. Run this file to get the continual csv communication started
	sam = SAMDynamicReader()
	sam.update_csv_file()

	### This is old test scripts, keep for personal documentation
	# frame = Image.open(sam.coredir + "/notebooks/videos/bedroom/0000" + str(0) + ".jpg").convert("RGB")
	# mask_combined = sam.initialize_from_frame(frame)
	# print(mask_combined[300, 300])
	# with torch.no_grad():
	# 	for i in range(1,9):
	# 		frame = Image.open(sam.coredir+"/0000"+str(i)+".jpg").convert("RGB")
	# 		mask_combined = sam.mask_step(frame, i)
	# 		print(mask_combined[300,300])
