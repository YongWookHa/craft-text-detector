from shapely.geometry import Polygon
import numpy as np
import torch
import cv2
import math

def get_device():
    "get device (CPU or GPU)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device

def order_points(box):

	x_sorted_arg = np.argsort(box[:, 0])
	if box[x_sorted_arg[0], 1] > box[x_sorted_arg[1], 1]:
		tl = x_sorted_arg[1]
	else:
		tl = x_sorted_arg[0]

	ordered_bbox = np.array([box[(tl + i) % 4] for i in range(4)])

	return ordered_bbox

def generate_word_bbox(
		character_heatmap, affinity_heatmap, character_threshold, affinity_threshold, word_threshold):

	"""
	Given the character heatmap, affinity heatmap, character and affinity threshold this function generates
	character bbox and word-bbox
	:param character_heatmap: Character Heatmap, numpy array, dtype=np.float32, shape = [height, width], value range [0, 1]
	:param affinity_heatmap: Affinity Heatmap, numpy array, dtype=np.float32, shape = [height, width], value range [0, 1]
	:param character_threshold: Threshold above which we say pixel belongs to a character
	:param affinity_threshold: Threshold above which we say a pixel belongs to a affinity
	:param word_threshold: Threshold of any pixel above which we say a group of characters for a word
	:return: {
		'word_bbox': word_bbox, type=np.array, dtype=np.int64, shape=[num_words, 4, 1, 2] ,
		'characters': char_bbox, type=list of np.array, dtype=np.int64, shape=[num_words, num_characters, 4, 1, 2] ,
		'affinity': affinity_bbox, type=list of np.array, dtype=np.int64, shape=[num_words, num_affinity, 4, 1, 2] ,
	}
	"""

	img_h, img_w = character_heatmap.shape

	""" labeling method """
	ret, text_score = cv2.threshold(character_heatmap, character_threshold, 1, 0)
	ret, link_score = cv2.threshold(affinity_heatmap, affinity_threshold, 1, 0)

	text_score_comb = np.clip(text_score + link_score, 0, 1)

	n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
		text_score_comb.astype(np.uint8),
		connectivity=4)

	det = []
	mapper = []
	for k in range(1, n_labels):

		try:
			# size filtering
			size = stats[k, cv2.CC_STAT_AREA]
			if size < 10:
				continue

			where = labels == k

			# thresholding
			if np.max(character_heatmap[where]) < word_threshold:
				continue

			# make segmentation map
			seg_map = np.zeros(character_heatmap.shape, dtype=np.uint8)
			seg_map[where] = 255
			seg_map[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area

			x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
			w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
			niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
			sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
			# boundary check
			if sx < 0:
				sx = 0
			if sy < 0:
				sy = 0
			if ex >= img_w:
				ex = img_w
			if ey >= img_h:
				ey = img_h
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
			seg_map[sy:ey, sx:ex] = cv2.dilate(seg_map[sy:ey, sx:ex], kernel)

			# make box
			np_contours = np.roll(np.array(np.where(seg_map != 0)), 1, axis=0).transpose().reshape(-1, 2)
			rectangle = cv2.minAreaRect(np_contours)
			box = cv2.boxPoints(rectangle)

			# align diamond-shape
			w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
			box_ratio = max(w, h) / (min(w, h) + 1e-5)
			if abs(1 - box_ratio) <= 0.1:
				l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
				t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
				box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

			# make clock-wise order
			start_idx = box.sum(axis=1).argmin()
			box = np.roll(box, 4 - start_idx, 0)
			box = np.array(box)

			det.append(box)
			mapper.append(k)

		except:
			# ToDo - Understand why there is a ValueError: math domain error in line
			#  niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)

			continue

	char_contours, _ = cv2.findContours(text_score.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	affinity_contours, _ = cv2.findContours(link_score.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	char_contours = link_to_word_bbox(char_contours, det)
	affinity_contours = link_to_word_bbox(affinity_contours, det)

	return {
		'word_bbox': np.array(det, dtype=np.int32).reshape([len(det), 4, 1, 2]),
		'characters': char_contours,
		'affinity': affinity_contours,
	}


def scale_bbox(contours, scale):

	mean = [np.array(i)[:, 0, :].mean(axis=0) for i in contours]
	centered_contours = [np.array(contours[i]) - mean[i][None, None, :] for i in range(len(contours))]
	scaled_contours = [centered_contours[i]*scale for i in range(len(centered_contours))]
	shifted_back = [scaled_contours[i] + mean[i][None, None, :] for i in range(len(scaled_contours))]
	shifted_back = [i.astype(np.int32) for i in shifted_back]

	return shifted_back


def link_to_word_bbox(to_find, word_bbox):

	if len(word_bbox) == 0:
		return [np.zeros([0, 4, 1, 2], dtype=np.int32)]
	word_sorted_character = [[] for _ in word_bbox]

	for cont_i, cont in enumerate(to_find):

		if cont.shape[0] < 4:
			continue

		rectangle = cv2.minAreaRect(cont)
		box = cv2.boxPoints(rectangle)

		if Polygon(box).area == 0:
			continue

		ordered_bbox = order_points(box)

		a = Polygon(cont.reshape([cont.shape[0], 2])).buffer(0)

		if a.area == 0:
			continue

		ratio = np.zeros([len(word_bbox)])

		# Polygon intersection usd for checking ratio

		for word_i, word in enumerate(word_bbox):
			b = Polygon(word.reshape([word.shape[0], 2])).buffer(0)
			ratio[word_i] = a.intersection(b).area/a.area

		word_sorted_character[np.argmax(ratio)].append(ordered_bbox)

	word_sorted_character = [
		np.array(word_i, dtype=np.int32).reshape([len(word_i), 4, 1, 2]) for word_i in word_sorted_character]

	return word_sorted_character


def generate_word_bbox_batch(
		batch_character_heatmap,
		batch_affinity_heatmap,
		character_threshold,
		affinity_threshold,
		word_threshold):

	"""
	Given the character heatmap, affinity heatmap, character and affinity threshold this function generates
	character bbox and word-bbox for the entire batch
	:param batch_character_heatmap: Batch Character Heatmap, numpy array, dtype=np.float32,
									shape = [batch_size, height, width], value range [0, 1]
	:param batch_affinity_heatmap: Batch Affinity Heatmap, numpy array, dtype=np.float32,
									shape = [batch_size, height, width], value range [0, 1]
	:param character_threshold: Threshold above which we say pixel belongs to a character
	:param affinity_threshold: Threshold above which we say a pixel belongs to a affinity
	:param word_threshold: Threshold above which we say a group of characters compromise a word
	:return: word_bbox
	"""

	word_bbox = []
	batch_size = batch_affinity_heatmap.shape[0]

	for i in range(batch_size):

		returned = generate_word_bbox(
			batch_character_heatmap[i],
			batch_affinity_heatmap[i],
			character_threshold,
			affinity_threshold,
			word_threshold)

		word_bbox.append(returned['word_bbox'])

	return word_bbox


def calc_iou(poly1, poly2):

	"""
	Function to calculate IOU of two bbox
	:param poly1: numpy array containing co-ordinates with shape [num_points, 1, 2] or [num_points, 2]
	:param poly2: numpy array containing co-ordinates with shape [num_points, 1, 2] or [num_points, 2]
	:return: float representing the IOU
	"""

	a = Polygon(poly1.reshape([poly1.shape[0], 2])).buffer(0)
	b = Polygon(poly2.reshape([poly2.shape[0], 2])).buffer(0)

	union_area = a.union(b).area

	if union_area == 0:
		return 0

	return a.intersection(b).area/union_area


def calculate_fscore(pred, target, text_target, unknown='###', text_pred=None, threshold=0.5):

	"""
	:param pred: numpy array with shape [num_words, 4, 2]
	:param target: numpy array with shape [num_words, 4, 2]
	:param text_target: list of the target text
	:param unknown: do not care text bbox
	:param text_pred: predicted text (Not useful in CRAFT implementation)
	:param threshold: overlap iou threshold over which we say the pair is positive
	:return:
	"""

	assert len(text_target) == target.shape[0], 'Some error in text target'

	if pred.shape[0] == target.shape[0] == 0:
		return {
			'f_score': 1.0,
			'precision': 1.0,
			'recall': 1.0,
			'false_positive': 0.0,
			'true_positive': 0.0,
			'num_positive': 0.0
		}

	if text_pred is None:
		check_text = False
	else:
		check_text = True

	already_done = np.zeros([len(target)], dtype=np.bool)

	false_positive = 0

	for no, i in enumerate(pred):

		found = False

		for j in range(len(target)):
			if already_done[j]:
				continue
			iou = calc_iou(i, target[j])
			if iou > threshold:
				if check_text:
					if text_pred[no] == text_target[j]:
						already_done[j] = True
						found = True
						break
				else:
					already_done[j] = True
					found = True
					break

		if not found:
			false_positive += 1

	if text_target is not None:
		true_positive = np.sum(already_done.astype(np.float32)[np.where(np.array(text_target) != unknown)[0]])
	else:
		true_positive = np.sum(already_done.astype(np.float32))

	if text_target is not None:
		num_positive = (np.where(np.array(text_target) != unknown)[0]).shape[0]
	else:
		num_positive = len(target)

	if true_positive == 0 and num_positive == 0:
		return {
			'f_score': 1.0,
			'precision': 1.0,
			'recall': 1.0,
			'false_positive': false_positive,
			'true_positive': true_positive,
			'num_positive': num_positive
		}

	if true_positive == 0:
		return {
			'f_score': 0.0,
			'precision': 0.0,
			'recall': 0.0,
			'false_positive': false_positive,
			'true_positive': true_positive,
			'num_positive': num_positive
		}

	precision = true_positive/(true_positive + false_positive)
	recall = true_positive / num_positive

	return {
		'f_score': 2*precision*recall/(precision + recall),
		'precision': precision,
		'recall': recall,
		'false_positive': false_positive,
		'true_positive': true_positive,
		'num_positive': num_positive
	}


def calculate_batch_fscore(pred, target, text_target, unknown='###', text_pred=None, threshold=0.5):

	"""
	Function to calculate the F-score of an entire batch. If lets say the model also predicted text,
	then a positive would be word_bbox IOU > threshold and exact text-match
	:param pred: list of numpy array having shape [num_words, 4, 2]
	:param target: list of numpy array having shape [num_words, 4, 2]
	:param text_target: list of target text, (not useful for CRAFT)
	:param text_pred: list of predicted text, (not useful for CRAFT)
	:param unknown: text specifying do not care scenario
	:param threshold: threshold value for iou above which we say a pair of bbox are positive
	:return:
	"""
	if text_target is None:
		text_target = [''.join(['_' for __ in range(len(target[_]))]) for _ in range(len(pred))]

	f_score = 0
	precision = 0
	recall = 0

	for i in range(len(pred)):
		if text_pred is not None:
			stats = calculate_fscore(pred[i], target[i], text_target[i], unknown, text_pred[i], threshold)
			f_score += stats['f_score']
			precision += stats['precision']
			recall += stats['recall']
		else:
			stats = calculate_fscore(pred[i], target[i], text_target[i], unknown, threshold=threshold)
			f_score += stats['f_score']
			precision += stats['precision']
			recall += stats['recall']

	return f_score/len(pred), precision/len(pred), recall/len(pred)