import cv2
import numpy as np
from ultralytics import YOLO
class ModelProcessor_front:
    def __init__(self, seg_model_path, kp_model_path, font_path="simhei.ttf"):
        # 加载分割模型和关键点模型
        self.seg_model = YOLO(seg_model_path)
        self.kp_model = YOLO(kp_model_path)
        self.font_path = font_path

    def draw_bboxes(self, image, bboxes):
        """绘制边界框"""
        for bbox in bboxes:
            x_min, y_min, x_max, y_max, conf = bbox[:5]
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
            cv2.putText(image, f"{conf:.2f}", (int(x_min), int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def predict_and_visualize_frame(self, frame):
        """
        对帧进行处理，返回处理后的帧。
        分割的轮廓会绘制回原图上，同时返回标注后的原图。
        如果 bbox 不存在，则直接返回原图。
        """
        # 关键点检测任务
        kp_results = self.kp_model(frame, conf=0.5)
        bbox_coords = []
        contours_with_colors = []

        # 创建一个空白的原图副本，用于绘制分割轮廓
        overlay_frame = frame.copy()

        for result in kp_results:
            if hasattr(result, 'boxes'):
                bboxes = result.boxes.data.cpu().numpy()

                # 如果没有检测到边界框，直接返回原始帧
                if len(bboxes) == 0:
                    return [],cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 遍历边界框
                for bbox in bboxes:
                    x_min, y_min, x_max, y_max, conf = bbox[:5]
                    bbox_coords.append((x_min, y_min, x_max, y_max))

        # 对每个检测区域进行分割任务和标注
        for idx, (x_min, y_min, x_max, y_max) in enumerate(bbox_coords):
            cropped_frame = frame[int(y_min):int(y_max), int(x_min):int(x_max)].copy()
            if cropped_frame.size == 0:
                continue  # 跳过空裁剪

            seg_results = self.seg_model(cropped_frame, conf=0.4, task="segment")
            masks = seg_results[0].masks
            classes = seg_results[0].names
            colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(len(seg_results[0].names))}
            if masks is None:
                continue
            for i, mask in enumerate(masks.data):
                class_id = int(seg_results[0].boxes.cls[i])
                class_name = classes[class_id]  # 获取类别名称
                mask = cv2.resize(mask.cpu().numpy().astype(np.uint8),
                                  (cropped_frame.shape[1], cropped_frame.shape[0]))

                # 提取轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    # 1. 将裁剪图上的轮廓点变换回原图上的坐标
                    scaled_contour = contour.copy()
                    scaled_contour[:, 0, 0] = contour[:, 0, 0] + int(x_min)
                    scaled_contour[:, 0, 1] = contour[:, 0, 1] + int(y_min)

                    # 2.将轮廓、颜色和类别信息存入列表
                    contours_with_colors.append({
                        'contour': scaled_contour,
                        'color': colors[class_id],  # 使用对应类别的颜色
                        'class_id': class_id,  # 类别 ID
                        'class_name': class_name  # 类别名称
                    })

                    # 2. 在原图副本（overlay_frame）上绘制轮廓
                    cv2.drawContours(overlay_frame, [scaled_contour], -1, colors[class_id], 2)

                # 在裁剪的图像上绘制轮廓（仅供参考）
                for contour in contours:
                    cv2.drawContours(cropped_frame, [contour], -1, colors[class_id], 2)

            # 在 cropped_frame 上绘制边界框
            cv2.rectangle(cropped_frame, (0, 0), (cropped_frame.shape[1], cropped_frame.shape[0]), (0, 255, 0), 2)
            cv2.putText(cropped_frame, f"Cropped Frame {idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 将结果返回：处理后的原图（包含分割轮廓）和最后的裁剪帧
        return contours_with_colors, cv2.cvtColor(
            cropped_frame if 'cropped_frame' in locals() else frame, cv2.COLOR_BGR2RGB)
class ModelProcessor_side:
    def __init__(self, seg_model_path, kp_model_path, font_path="simhei.ttf"):
        # 加载分割模型和关键点模型
        self.seg_model = YOLO(seg_model_path)
        self.kp_model = YOLO(kp_model_path)
        self.font_path = font_path

    def draw_bboxes(self, image, bboxes):
        """绘制边界框"""
        for bbox in bboxes:
            x_min, y_min, x_max, y_max, conf = bbox[:5]
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
            cv2.putText(image, f"{conf:.2f}", (int(x_min), int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def predict_and_visualize_frame(self, frame):
        """
        对帧进行处理，返回处理后的帧。
        分割的轮廓会绘制回原图上，同时返回标注后的原图。
        如果 bbox 不存在，则直接返回原图。
        """
        # 关键点检测任务
        kp_results = self.kp_model(frame, conf=0.5)
        bbox_coords = []
        contours_with_colors = []

        # 创建一个空白的原图副本，用于绘制分割轮廓
        overlay_frame = frame.copy()

        for result in kp_results:
            if hasattr(result, 'boxes'):
                bboxes = result.boxes.data.cpu().numpy()

                # 如果没有检测到边界框，直接返回原始帧
                if len(bboxes) == 0:
                    return [],cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 遍历边界框
                for bbox in bboxes:
                    x_min, y_min, x_max, y_max, conf = bbox[:5]
                    bbox_coords.append((x_min, y_min, x_max, y_max))

        # 对每个检测区域进行分割任务和标注
        for idx, (x_min, y_min, x_max, y_max) in enumerate(bbox_coords):
            cropped_frame = frame[int(y_min):int(y_max), int(x_min):int(x_max)].copy()
            if cropped_frame.size == 0:
                continue  # 跳过空裁剪
            seg_results = self.seg_model(cropped_frame, conf=0.4, task="segment")
            masks = seg_results[0].masks
            classes = seg_results[0].names
            colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(len(seg_results[0].names))}
            if masks is None:
                continue
            for i, mask in enumerate(masks.data):
                class_name = classes[class_id]  # 获取类别名称
                class_id = int(seg_results[0].boxes.cls[i])
                mask = cv2.resize(mask.cpu().numpy().astype(np.uint8),
                                  (cropped_frame.shape[1], cropped_frame.shape[0]))

                # 提取轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    # 1. 将裁剪图上的轮廓点变换回原图上的坐标
                    scaled_contour = contour.copy()
                    scaled_contour[:, 0, 0] = contour[:, 0, 0] + int(x_min)
                    scaled_contour[:, 0, 1] = contour[:, 0, 1] + int(y_min)

                    # 2.将轮廓、颜色和类别信息存入列表
                    contours_with_colors.append({
                        'contour': scaled_contour,
                        'color': colors[class_id],  # 使用对应类别的颜色
                        'class_id': class_id,  # 类别 ID
                        'class_name': class_name  # 类别名称
                    })

                    # 2. 在原图副本（overlay_frame）上绘制轮廓
                    cv2.drawContours(overlay_frame, [scaled_contour], -1, colors[class_id], 2)

                # 在裁剪的图像上绘制轮廓（仅供参考）
                for contour in contours:
                    cv2.drawContours(cropped_frame, [contour], -1, colors[class_id], 2)
            # 在 cropped_frame 上绘制边界框
            cv2.rectangle(cropped_frame, (0, 0), (cropped_frame.shape[1], cropped_frame.shape[0]), (0, 255, 0), 2)
            cv2.putText(cropped_frame, f"Cropped Frame {idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # 将结果返回：处理后的原图（包含分割轮廓）和最后的裁剪帧
        return contours_with_colors, cv2.cvtColor(
            cropped_frame if 'cropped_frame' in locals() else frame, cv2.COLOR_BGR2RGB)