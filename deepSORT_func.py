from colorama import Fore, Back, Style, init
init(autoreset=True)
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from utils.general import xyxy2xywh
from utils.plots import Annotator, colors
import bbox as BOX
import torch
import numpy as np

class deepSORT:
		
	def __init__(self, device, names):
		deepSort_Config = "/data/codes/30_yolo/yolov5/deep_sort/configs/deep_sort.yaml"
		deepSort_model = "resnet50_MSMT17"
  
		self.names = names

		cfg = get_config()
		cfg.merge_from_file(deepSort_Config)
		self.deepsort = DeepSort(deepSort_model,
							device,
							max_dist=cfg.DEEPSORT.MAX_DIST,
							max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
							max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
							)
  
		
	def update(self, det, det_lc, im0, verbose=0):
     
		self.annotator = Annotator(im0, line_width=2, pil=not ascii)
  
		xywhs = xyxy2xywh(det[:, 0:4])
		confs = det[:, 4]
		clss = det[:, 5]		
		trks_confirm, trk_missed, trk_new = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

		
		if verbose: self.print_tracks(det,trks_confirm,trk_new,trk_missed)
		if verbose: self.annotate_tracks(det,trks_confirm,trk_new,trk_missed)
   
		tum, tm, tyolo = self.match_missed(trk_missed,trks_confirm,trk_new, det_lc, im0, debug=False)
  
		im0 = self.annotator.result()
		return trks_confirm, trk_missed, trk_new, im0
	
 
 	
	def annotate_boxes(self,annotator, det, names, istensor=False, color=(255,0,0), pad=0):
		if istensor: det = det.cpu().detach().numpy()
		if len(det[0])>6:
			boxes, confs, classes, tid = det[:, 0:4], det[:,4], det[:,5], det[:,6]
			for j, (box, conf, cls, trk) in enumerate(zip(boxes, confs, classes, tid)):
				# label = f'{int(cls)} c:{conf/10000:.2f} t:{trk}'
				label = f'{names[int(cls)]} conf:{conf/10000:.2f} track:{trk}'
				box[0],box[1],box[2],box[3] = box[0]-pad,box[1]-pad,box[2]+pad*2,box[3]+pad*2
				annotator.box_label(box, label, color)
		else:
			boxes, confs, classes = det[:, 0:4], det[:,4], det[:,5]
			for j, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
				box[0],box[1],box[2],box[3] = box[0]-pad,box[1]-pad,box[2]+pad*2,box[3]+pad*2
				# label = f'{int(cls)} c:{conf:.2f}'
				label = f'{names[int(cls)]} conf:{conf:.2f}'
				annotator.box_label(box, label, color)
				
			
   
   	
	def match_boxes(self,boxes1, boxes2, iou_thresh = 0.80):

		if torch.is_tensor(boxes1): boxes1 = boxes1.cpu().detach().numpy()
		if torch.is_tensor(boxes2): boxes2 = boxes2.cpu().detach().numpy()
		
		boxes_boxes1 = BOX.BBox2DList(boxes1[:,:4])       # X1, Y1, X2, Y2, Conf, Cls, TID
		boxes_boxes2 = BOX.BBox2DList(boxes2[:,:4])       # X1, Y1, X2, Y2, Conf, Cls, TID
		
		conf_boxes1  = boxes1[:,4]/10000
		conf_boxes2  = boxes2[:,4]
		
		class_boxes1 = boxes1[:,5]
		class_boxes2 = boxes2[:,5]

		matched_idx = []
		matched1, unmatched1 = [], []
		matched2, unmatched2 = [], []
		
		IoUs = BOX.metrics.multi_iou_2d(boxes_boxes1, boxes_boxes2)
		iou = np.logical_and(IoUs>iou_thresh,IoUs<1)
		
		len_box1,len_box2 = iou.shape
		
		for i in range(len_box1):
			# iiou = iou[i] if iou.shape[1]==1 else iou[:,i]
			# if iiou.any():
			if iou[i].any():
				matched1.append(boxes1[i])
			else:
				unmatched1.append(boxes1[i])
		
		for j in range(len_box2):
			# iiou = iou[j] if iou.shape[1]==1 else iou[:,j]
			# if iiou.any():
			if iou[:,j].any():
				matched2.append(boxes2[j])
			else:
				unmatched2.append(boxes2[j])
		
				
				# for j in range(len_box2):
				#     if iiiou: 
				#         matched_idx.append([i,j])
				# for j,iiiou in enumerate(iiou):
				#     if not iiiou:
			
		
		return np.array(matched1), np.array(unmatched1), np.array(matched2), np.array(unmatched2), np.array(matched_idx)
		



	def match_missed(self,trk_missed,trks_confirm,trk_new,lc_boxes, im0, debug=False):
		if torch.is_tensor(lc_boxes): lc_boxes = lc_boxes.cpu().detach().numpy() 
		if len(trk_missed) > 0 and len(lc_boxes) > 0:
			unmatched_trk_missed = trk_missed
			if debug: print("\n\nMissed Tracks\n", trk_missed)
			
			if len(unmatched_trk_missed) > 0 and len(trks_confirm) > 0:
				matched_trk_missed, unmatched_trk_missed, matched_trk_confirm, unmatched_trk_confirm, matched_idx = \
					self.match_boxes(unmatched_trk_missed, trks_confirm, iou_thresh = 0.70)
				if debug: print("\nUnmatched Missed vs DeepSORT Tracks\n", unmatched_trk_missed)
					
			if len(unmatched_trk_missed) > 0 and len(trk_new) > 0:
				matched_trk_missed, unmatched_trk_missed, matched_trk_new, unmatched_trk_new, matched_idx = \
					self.match_boxes(unmatched_trk_missed, trk_new, iou_thresh = 0.70)
				if debug: print("\nUnmatched Missed vs DeepSORT+New Tracks\n", unmatched_trk_missed)

			# lc_boxes[:,:4] = [self.deepsort._xywh_to_xyxy(box) for box in lc_boxes[:,:4]]
			# lc_boxes[:,:4] = deepsort._xywh_to_xyxy(lc_boxes[:,0:4]) 

			if len(unmatched_trk_missed) > 0 and len(lc_boxes) > 0:
				matched_trk_missed, unmatched_trk_missed, matched_det_yolo_lc, unmatched_det_yolo_lc, matched_idx = \
					self.match_boxes(unmatched_trk_missed, lc_boxes, iou_thresh = 0.70)
				debug = True				
				if debug: print("\nUnmatched Missed vs DeepSORT+New+LC Yolo Tracks\n", unmatched_trk_missed)
				if debug: print("\nMatched Remaining Missed\n", matched_trk_missed)
				if debug: print("\nMatched LC Yolo Tracks\n", matched_det_yolo_lc)
				debug = False
			
				if not len(matched_det_yolo_lc)==0:
					self.annotate_boxes(self.annotator, matched_det_yolo_lc, self.names, istensor=torch.is_tensor(matched_det_yolo_lc), color=(0,125,255), pad=2) 
					self.deepsort, count = self.update_missed_box(self.deepsort,matched_trk_missed, matched_det_yolo_lc, im0, count)
					# if save_txt:
					# 	save_results(matched_trk_missed,txt_path,im0) # Save matched missed-lc boxes
					# 	count_saves = count_saves + len(matched_trk_missed)
			return unmatched_trk_missed, matched_trk_missed, matched_det_yolo_lc
     
	def print_tracks(self,det,trks_confirm,trk_new,trk_missed):
		print(Style.BRIGHT + f"Tracks",end="    ")
		print(Style.BRIGHT + Fore.GREEN + f"Yolo-v5:{len(det)}",end="    ")
		print(Style.BRIGHT + Fore.BLUE + f"DeepSORT:{len(trks_confirm)}",end="    ")
		print(Style.BRIGHT + Fore.MAGENTA + f"Modified:{len(trks_confirm)+len(trk_new)}",end="    ")
		print(Style.BRIGHT + Fore.RED + f"Missed:{len(trk_missed)}")
  	
	def annotate_tracks(self,det,trks_confirm,trk_new,trk_missed):
		if not len(trks_confirm)==0:
			self.annotate_boxes(self.annotator, trks_confirm, self.names, color=(255,0,0))		# Confirmed - Blue
		if not len(trk_missed)==0:
				self.annotate_boxes(self.annotator, trk_missed, self.names, color=(0,0,255))		# Missed - Red
		if not len(trk_new)==0:
			self.annotate_boxes(self.annotator, trk_new, self.names, color=(255,0,255))	  # New Tracks - Purple
   