import math

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont


class ModelProcessor:
    def __init__(self, seg_model_path, kp_model_path=None, font_path="simhei.ttf"):
        # 加载分割模型和关键点模型
        self.seg_model = YOLO(seg_model_path)
        self.kp_model = YOLO(kp_model_path)
        self.font_path = font_path
        # 初始化关键点信息，仅对部分关键点进行定义
        self.keypoint_info =  {
                 5: {'name': 'shenmen', 'color': (255, 0, 0), 'region': 'sanjiaowoqu', 'ratio': 0.15},
                 8: {'name': 'naogan', 'color': (0, 255, 0), 'region': 'duierpingqu', 'ratio': 0.1},
                 9: {'name': 'yuanzhong', 'color': (0, 0, 255), 'region': 'duierpingqu', 'ratio': 0.1},
                 13: {'name': 'chuiqian', 'color': (255, 255, 0), 'region': 'erchuiqu', 'ratio': 0.15}
               }

    def draw_keypoints(self, image, keypoints, conf_threshold=0.5):
        """绘制所有关键点及中文名称"""
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.truetype(self.font_path, 20)
        for idx, kp in enumerate(keypoints):
            if len(kp) >= 3:
                x, y, conf = kp[:3]
                if conf > conf_threshold:
                    info = self.keypoint_info.get(idx, {'name': f'未知{idx}', 'color': (255, 255, 255)})
                    name, color = info['name'], info['color']
                    radius = 10
                    draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=color, outline=color)
                    draw.text((x + 15, y - 15), name, fill=color, font=font)
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    def draw_selected_keypoints(self, image, keypoints, allowed_indices={5, 8, 9, 13},
                                conf_threshold=0.25, dot_radius=20, font_size=60, text_offset=5):
        """
        绘制关键点，仅绘制 allowed_indices 内的关键点
        • dot_radius: 关键点圆形的半径（点的大小）
        • font_size: 字体大小
        • text_offset: 文本与圆形的间隔距离，确保文本不会被圆遮挡
        """
        # 将 BGR 图像转换为 PIL Image 进行绘制
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.truetype(self.font_path, font_size)

        for idx, kp in enumerate(keypoints):
            if idx not in allowed_indices:
                continue
            if len(kp) >= 3:
                x, y, conf = kp[:3]
                if conf > conf_threshold:
                    # 读取关键点信息（中文名称、颜色）
                    info = self.keypoint_info.get(idx, {'name': f'未知{idx}', 'color': (255, 255, 255)})
                    name, color = info['name'], info['color']
                    # 绘制圆形：中心为 (x, y)，半径为 dot_radius
                    left_up = (x - dot_radius, y - dot_radius)
                    right_down = (x + dot_radius, y + dot_radius)
                    draw.ellipse([left_up, right_down], fill=color, outline=color)
                    # 绘制文本：设定位置为圆形右侧并稍微偏上一点，避免被图形遮挡
                    text_position = (x + dot_radius + text_offset, y - dot_radius)
                    draw.text(text_position, name, fill=color, font=font)

        # 返回绘制完毕后的图像，转换为 BGR 格式
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    def draw_bboxes(self, image, bboxes):
        """在图像上绘制边界框"""
        for bbox in bboxes:
            x_min, y_min, x_max, y_max, conf = bbox[:5]
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
            cv2.putText(image, f"{conf:.2f}", (int(x_min), int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def predict_and_visualize(self, image_path, output_path):
        """
        原有功能：
        1. 关键点检测后在图像上绘制关键点与边界框；
        2. 对每个检测区域执行分割任务，将分割轮廓绘制回原图；
        3. 最后保存包含分割及关键点标注的结果图像到 output_path。
        """
        image = cv2.imread(image_path)
        assert image is not None, f"无法读取图像：{image_path}"

        kp_results = self.kp_model(image, conf=0.25)
        cropped_images = []
        bbox_coords = []
        for result in kp_results:
            if hasattr(result, 'boxes'):
                bboxes = result.boxes.data.cpu().numpy()
                self.draw_bboxes(image, bboxes)
                for bbox in bboxes:
                    x_min, y_min, x_max, y_max, conf = bbox[:5]
                    bbox_coords.append((x_min, y_min, x_max, y_max))
                    cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
                    cropped_images.append(cropped_image)
            if hasattr(result, 'keypoints'):
                keypoints = result.keypoints.data.cpu().numpy()
                for person_keypoints in keypoints:
                    image = self.draw_keypoints(image, person_keypoints)
        for idx, cropped_image in enumerate(cropped_images):
            if cropped_image.size == 0:
                continue
            seg_results = self.seg_model(cropped_image, conf=0.4, task="segment")
            masks = seg_results[0].masks
            classes = seg_results[0].names
            colors = {i: tuple(np.random.randint(0, 255, 3).tolist())
                      for i in range(len(classes))}
            x_min, y_min, x_max, y_max = bbox_coords[idx]
            for i, mask in enumerate(masks.data):
                class_id = int(seg_results[0].boxes.cls[i])
                mask = cv2.resize(mask.cpu().numpy().astype(np.uint8),
                                  (int(x_max - x_min), int(y_max - y_min)))
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    contour[:, 0, 0] += int(x_min)
                    contour[:, 0, 1] += int(y_min)
                    cv2.drawContours(image, [contour], -1, colors[class_id], 8)
        cv2.imwrite(output_path, image)
        # print(f"结果已保存至 {output_path}")

    def predict_and_visualize_frame(self, frame, line_thickness=2):
        """
        针对视频帧的原有处理：
        1. 检测边界框；
        2. 针对每个检测区域切割区域（crop），执行分割任务并绘制分割轮廓到一个副本上；
        3. 返回分割轮廓信息和最后一次 crop 的图像（RGB格式），若无 crop 则返回原始帧。
        """
        kp_results = self.kp_model(frame, conf=0.25)
        bbox_coords = []
        contours_with_colors = []
        overlay_frame = frame.copy()
        for result in kp_results:
            if hasattr(result, 'boxes'):
                bboxes = result.boxes.data.cpu().numpy()
                if len(bboxes) == 0:
                    return [], cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for bbox in bboxes:
                    x_min, y_min, x_max, y_max, conf = bbox[:5]
                    bbox_coords.append((x_min, y_min, x_max, y_max))
        # 执行分割任务，将结果绘制在 overlay_frame 上
        for idx, (x_min, y_min, x_max, y_max) in enumerate(bbox_coords):
            cropped_frame = frame[int(y_min):int(y_max), int(x_min):int(x_max)].copy()
            if cropped_frame.size == 0:
                continue
            seg_results = self.seg_model(cropped_frame, conf=0.4, task="segment")
            classes = seg_results[0].names
            masks = seg_results[0].masks
            if masks is None:
                continue
            colors = {i: tuple(np.random.randint(0, 255, 3).tolist())
                      for i in range(len(classes))}
            for i, mask in enumerate(masks.data):
                class_id = int(seg_results[0].boxes.cls[i])
                mask = cv2.resize(mask.cpu().numpy().astype(np.uint8),
                                  (cropped_frame.shape[1], cropped_frame.shape[0]))
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    scaled_contour = contour.copy()
                    scaled_contour[:, 0, 0] = contour[:, 0, 0] + int(x_min)
                    scaled_contour[:, 0, 1] = contour[:, 0, 1] + int(y_min)
                    contours_with_colors.append({
                        'contour': scaled_contour,
                        'color': colors[class_id],
                        'class_id': class_id,
                        'class_name': classes[class_id]
                    })
                    cv2.drawContours(overlay_frame, [scaled_contour], -1, colors[class_id], 2)
        # 若检测到裁剪区域，则返回最后一次 crop 的（已绘制）图像，否则返回原始帧
        final_crop = None
        if len(bbox_coords) > 0:
            x_min, y_min, x_max, y_max = bbox_coords[-1]
            final_crop = overlay_frame[int(y_min):int(y_max), int(x_min):int(x_max)].copy()
        else:
            final_crop = frame.copy()
        return contours_with_colors, cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB)

    def predict_and_visualize_frame_seg_and_selected_keypoints(self, frame, line_thickness=2):
        """
        新增函数功能：
        1. 针对输入帧先进行边界框和关键点（仅保留索引 {5,8,9,13}）检测，
           将边界框和关键点绘制到绘制副本 processed_frame 上；
        2. 对每个检测到的边界框区域（区域位置来源于边界框）执行分割任务，
           并将分割结果（轮廓）绘制到 processed_frame 上；
        3. 最后，取 processed_frame 对应最后一个边界框区域的裁剪结果作为 cropimg 输出，
           此 cropimg 为画好所有检测结果与分割轮廓的图像（RGB格式）。
        """
        # 使用副本保存绘制结果
        processed_frame = frame.copy()
        kp_results = self.kp_model(frame, conf=0.25)
        bbox_coords = []
        contours_with_colors = []
        # 遍历检测结果：边界框及关键点（仅选取索引 {5,8,9,13}）
        for result in kp_results:
            if hasattr(result, 'boxes'):
                bboxes = result.boxes.data.cpu().numpy()
                if len(bboxes) == 0:
                    return [], cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for bbox in bboxes:
                    x_min, y_min, x_max, y_max, conf = bbox[:5]
                    bbox_coords.append((x_min, y_min, x_max, y_max))
                    # 绘制边界框到 processed_frame
                    cv2.rectangle(processed_frame, (int(x_min), int(y_min)),
                                  (int(x_max), int(y_max)), (0, 0, 255), 2)
                    cv2.putText(processed_frame, f"{conf:.2f}", (int(x_min), int(y_min) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if hasattr(result, 'keypoints'):
                keypoints = result.keypoints.data.cpu().numpy()
                for person_keypoints in keypoints:
                    processed_frame = self.draw_selected_keypoints(processed_frame,
                                                                   person_keypoints,
                                                                   allowed_indices={5, 8, 9, 13})
        # 针对每个边界框区域执行分割任务，将分割轮廓绘制到 processed_frame 上
        for idx, (x_min, y_min, x_max, y_max) in enumerate(bbox_coords):
            # 以原始 frame 对应边界框区域进行分割，但绘制在 processed_frame 上
            cropped_area = frame[int(y_min):int(y_max), int(x_min):int(x_max)].copy()
            if cropped_area.size == 0:
                continue
            seg_results = self.seg_model(cropped_area, conf=0.4, task="segment")
            classes = seg_results[0].names
            masks = seg_results[0].masks
            if masks is None:
                continue
            colors = {i: tuple(np.random.randint(0, 255, 3).tolist())
                      for i in range(len(classes))}
            for i, mask in enumerate(masks.data):
                class_id = int(seg_results[0].boxes.cls[i])
                mask = cv2.resize(mask.cpu().numpy().astype(np.uint8),
                                  (int(x_max - x_min), int(y_max - y_min)))
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    # 调整坐标：将裁剪区域的坐标转换到整幅图中
                    contour[:, 0, 0] += int(x_min)
                    contour[:, 0, 1] += int(y_min)
                    contours_with_colors.append({
                        'contour': contour,
                        'color': colors[class_id],
                        'class_id': class_id,
                        'class_name': classes[class_id]
                    })
                    cv2.drawContours(processed_frame, [contour], -1, colors[class_id], 6)
        # 按照接口要求，cropimg 应为最终绘制结果对应最后一个边界框区域的裁剪图
        final_crop = None
        if len(bbox_coords) > 0:
            fx_min, fy_min, fx_max, fy_max = bbox_coords[-1]
            final_crop = processed_frame[int(fy_min):int(fy_max), int(fx_min):int(fx_max)].copy()
        else:
            final_crop = processed_frame.copy()
        return contours_with_colors, cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB)

    def predict_and_visualize_frame_seg_and_selected_keypoints_with_region(self,frame=None,default_line_thickness=2):
        """
        基于 predict_and_visualize_frame_seg_and_selected_keypoints 的实现，
        新增如下功能：
          • 对于已检测出的关键点（索引 {5,8,9,13}），利用 self.keypoint_info 中定义的所属区域（region）和比例（ratio），
            在分割结果中查找对应类别的轮廓（即轮廓的类别名等于 keypoint_info 中的 region 字段）；如果有多个，则选择面积最小的那个；
          • 计算该轮廓面积，然后按设定比例计算新形状（以圆形为例）的面积，新形状面积 = ratio * 分割轮廓面积，
          • 根据面积求出圆形半径（面积 = π * r^2），以关键点为中心绘制此圆，并将对应轮廓信息（颜色、关键点名称等）保存到 contours_with_colors 中，
          • 同时将圆形轮廓绘制到 processed_frame 上。
        参数：
          frame: 输入图像（BGR格式），若传入 None 则需要外部保证 frame 有值
          line_thickness: 分割轮廓及后续形状的绘制线宽（默认值为 2）
        返回：
          contours_with_colors 列表，以及最后一次检测边界框区域（裁剪后）对应的图像（RGB格式）
        注意：
          1. 请确保 self.keypoint_info 中对各关键点已增加如下字段，例如：
               {
                 5: {'name': '神门', 'color': (255, 0, 0), 'region': 'sanjiaowoqu', 'ratio': 0.5},
                 8: {'name': '脑干', 'color': (0, 255, 0), 'region': 'duierpingqu', 'ratio': 0.6},
                 9: {'name': '缘中', 'color': (0, 0, 255), 'region': 'duierpingqu', 'ratio': 0.7},
                 13: {'name': '垂前', 'color': (255, 255, 0), 'region': 'erchuiqu', 'ratio': 0.8}
               }
          2. 分割模型返回的轮廓字典中，键 'class_name' 表示该分割轮廓所属类别，需与 keypoint_info 中的 region 值对应。
        """
        import math
        # 若外部未传入 frame，则需保证已定义
        if frame is None:
            raise ValueError("必须传入有效的 frame 图像")
        processed_frame = frame.copy()
        kp_results = self.kp_model(frame, conf=0.25)
        bbox_coords = []
        contours_with_colors = []
        selected_keypoints = []  # 列表中保存关键点的位置信息和对应索引
        allowed_indices = {5, 8, 9, 13}

        # 遍历检测结果：绘制边界框和关键点（只保留 allowed_indices 内的关键点）
        for result in kp_results:
            if hasattr(result, 'boxes'):
                bboxes = result.boxes.data.cpu().numpy()
                if len(bboxes) == 0:
                    return [], cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for bbox in bboxes:
                    x_min, y_min, x_max, y_max, conf = bbox[:5]
                    bbox_coords.append((x_min, y_min, x_max, y_max))
                    cv2.rectangle(processed_frame, (int(x_min), int(y_min)),
                                  (int(x_max), int(y_max)), (0, 0, 255), default_line_thickness)
                    cv2.putText(processed_frame, f"{conf:.2f}", (int(x_min), int(y_min) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), default_line_thickness)
            if hasattr(result, 'keypoints'):
                keypoints = result.keypoints.data.cpu().numpy()
                for person_keypoints in keypoints:
                    # 调用已有函数绘制选定关键点
                    processed_frame = self.draw_selected_keypoints(processed_frame,
                                                                   person_keypoints,
                                                                   allowed_indices=allowed_indices)
                    # 记录每个允许关键点的信息
                    for idx, kp in enumerate(person_keypoints):
                        if idx not in allowed_indices:
                            continue
                        if len(kp) >= 3:
                            x, y, conf = kp[:3]
                            if conf > 0.25:
                                info = self.keypoint_info.get(idx, {'name': f'未知{idx}'})
                                selected_keypoints.append({
                                    'index': idx,
                                    'x': x,
                                    'y': y,
                                    'name': info.get('name', f'未知{idx}'),
                                    'region': info.get('region', None),
                                    'ratio': info.get('ratio', 0.5)  # 默认比例可以调整
                                })
        # 对每个边界框区域执行分割任务，将分割轮廓绘制到 processed_frame 上
        for (x_min, y_min, x_max, y_max) in bbox_coords:
            cropped_area = frame[int(y_min):int(y_max), int(x_min):int(x_max)].copy()
            if cropped_area.size == 0:
                continue
            seg_results = self.seg_model(cropped_area, conf=0.4, task="segment")
            classes = seg_results[0].names
            masks = seg_results[0].masks
            if masks is None:
                continue
            # 为每个类别随机生成颜色（也可以根据需要固定颜色）
            colors = {i: tuple(np.random.randint(0, 255, 3).tolist())
                      for i in range(len(classes))}
            for i, mask in enumerate(masks.data):
                class_id = int(seg_results[0].boxes.cls[i])
                mask = cv2.resize(mask.cpu().numpy().astype(np.uint8),
                                  (int(x_max - x_min), int(y_max - y_min)))
                cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for contour in cnts:
                    # 将裁剪区域中的轮廓坐标调整到原图位置
                    contour[:, 0, 0] += int(x_min)
                    contour[:, 0, 1] += int(y_min)
                    # 保存轮廓信息，其中 contour 的 'class_name' 从分割模型返回的类别名称决定
                    contours_with_colors.append({
                        'contour': contour,
                        'color': colors[class_id],
                        'class_id': class_id,
                        'class_name': classes[class_id]
                    })
                    cv2.drawContours(processed_frame, [contour], -1, colors[class_id], default_line_thickness)
        # 针对每个允许关键点，直接使用其在 keypoint_info 中定义的所属区域，查找分割结果中对应类别的轮廓
        for kp in selected_keypoints:
            kp_x, kp_y = int(kp['x']), int(kp['y'])
            region_label = kp.get('region', None)
            ratio = kp.get('ratio', 0.5)
            if region_label is None:
                continue
            # 在之前保存的分割轮廓中寻找类别名等于 region_label 的轮廓
            candidate_contours = []
            for item in contours_with_colors:
                if item.get('class_name', '') == region_label:
                    candidate_contours.append(item)
            if len(candidate_contours) == 0:
                print(region_label)
                print('没找到对应区域的轮廓')
                # 没有找到对应区域的轮廓，则跳过该关键点
                continue
            # 如果找到多个，选择面积最小的轮廓
            selected_item = max(candidate_contours, key=lambda d: cv2.contourArea(d['contour']))
            base_area = cv2.contourArea(selected_item['contour'])
            # 根据预设比例计算新形状面积
            new_area = ratio * base_area
            # 以圆形为例：面积 = π*r^2 -> r = sqrt(new_area/π)
            radius = int(math.sqrt(new_area / math.pi))
            # 根据关键点中心和计算半径构造圆形轮廓（360 度分 36 个点）
            circle_points = []
            for angle in range(0, 360, 10):
                theta = math.radians(angle)
                x_pt = kp_x + int(radius * math.cos(theta))
                y_pt = kp_y + int(radius * math.sin(theta))
                circle_points.append([x_pt, y_pt])
            new_contour = np.array(circle_points, dtype=np.int32).reshape((-1, 1, 2))
            # 使用关键点对应定义的颜色绘制新的圆形轮廓
            kp_color = self.keypoint_info.get(kp['index'], {}).get('color', (0, 255, 255))
            cv2.drawContours(processed_frame, [new_contour], -1, kp_color, default_line_thickness)
            # 将该新生成的轮廓添加到结果中，class_id 可任意设置，这里使用 -1；class_name 使用关键点命名
            contours_with_colors.append({
                'contour': new_contour,
                'color': kp_color,
                'class_id': -1,
                'class_name': kp['name']
            })
        # 最后，若检测到边界框，则将最后一个边界框区域裁剪后作为最终输出，否则输出整个 processed_frame
        if len(bbox_coords) > 0:
            fx_min, fy_min, fx_max, fy_max = bbox_coords[-1]
            final_crop = processed_frame[int(fy_min):int(fy_max), int(fx_min):int(fx_max)].copy()
        else:
            final_crop = processed_frame.copy()
        return contours_with_colors, cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB)

    def draw_selected_segmentation_contours_and_crop(self, image, seg_conf_threshold=0.4,
                                                     kp_conf_threshold=0.25, line_thickness=2):
        """
        1. 使用分割模型对图像进行分割，并只绘制预设分割类别（用拼音表示）的轮廓。
        2. 使用 kpmodel 对图像进行检测，获取边界框；
        3. 最后对绘制了分割轮廓的图像，以 kpmodel 检测到的最后一个边界框进行裁剪。

        参数：
           image: BGR 格式的输入图像
           seg_conf_threshold: 分割推理的置信度阈值，默认 0.4
           kp_conf_threshold: kpmodel 关键点检测的置信度阈值，默认 0.25
           line_thickness: 绘制轮廓和边界框的线条宽度

        返回：
           裁剪后的图像（RGB 格式），以及完整绘制有分割轮廓和边界框的图像（BGR 格式）。
        """
        # --- 调用 kpmodel 获得边界框信息 ---
        kp_results = self.kp_model(image, conf=kp_conf_threshold)
        bbox_coords = []
        for result in kp_results:
            if hasattr(result, 'boxes'):
                bboxes = result.boxes.data.cpu().numpy()
                for box in bboxes:
                    # 假定 box 为 [x_min, y_min, x_max, y_max, conf]
                    x_min, y_min, x_max, y_max, conf = box[:5]
                    bbox_coords.append((x_min, y_min, x_max, y_max))
                    # 同时也可以在图像上绘制边界框
                    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255),
                                  line_thickness)
                    cv2.putText(image, f"{conf:.2f}", (int(x_min), int(y_min) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), line_thickness)

        # --- 绘制指定类别的分割轮廓 ---
        # 定义允许的类别（拼音名称）
        allowed_categories = {
            "erchuiqu",  # 耳垂区
            "duierlunqu",  # 对耳轮区
            "duierpingqu",  # 对耳屏区
            "erzhouqu",  # 耳周区
            "erpingqu",  # 耳屏区
            "erjiaqu",  # 耳甲区
            "sanjiaowoqu",  # 三角窝区
            "erlunqu"
        }
        # 定义颜色映射
        color_map = {
            "erchuiqu": (255, 0, 0),
            "duierlunqu": (0, 255, 0),
            "duierpingqu": (0, 0, 255),
            "erzhouqu": (255, 255, 0),
            "erpingqu": (255, 0, 255),
            "erjiaqu": (0, 255, 255),
            "sanjiaowoqu": (128, 128, 0),
            "erlunqu":(0,128,128)
        }
        # 执行分割推理
        seg_results = self.seg_model(image, conf=seg_conf_threshold, task="segment")
        if len(seg_results) > 0:
            result = seg_results[0]
            if hasattr(result, 'masks') and result.masks is not None:
                classes = result.names  # 假定为 {class_id: class_name} 格式，且名称为拼音
                masks = result.masks.data
                # 如果模型有 boxes 信息，则使用 boxes.cls 确定类别 id，否则循环
                if hasattr(result, 'boxes'):
                    cls_array = result.boxes.cls.cpu().numpy()
                else:
                    cls_array = list(range(len(masks)))
                orig_h, orig_w = image.shape[:2]
                for i in range(len(masks)):
                    class_id = int(cls_array[i])
                    class_name = classes[class_id]
                    if class_name not in allowed_categories:
                        continue
                    # 将 mask 转换为 uint8，并调整到原图尺寸（若不一致）
                    mask = masks[i].cpu().numpy().astype(np.uint8)
                    mask_h, mask_w = mask.shape[:2]
                    if (mask_h, mask_w) != (orig_h, orig_w):
                        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    # 查找轮廓
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        color = color_map.get(class_name, (0, 255, 255))
                        cv2.drawContours(image, [contour], -1, color, line_thickness)
                        # 在轮廓附近标注类别名称（选取轮廓第一个点）
                        if len(contour) > 0:
                            pt = tuple(contour[0][0])
                            cv2.putText(image, class_name, pt, cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8, color, line_thickness, cv2.LINE_AA)

        # --- 裁剪最终图像 ---
        # 如果 kpmodel 检测到了边界框，则使用最后一个边界框对图像进行裁剪
        if len(bbox_coords) > 0:
            x_min, y_min, x_max, y_max = bbox_coords[-1]
            # 裁剪，并转换为 RGB 格式
            crop_img = image[int(y_min):int(y_max), int(x_min):int(x_max)].copy()
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        else:
            crop_img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

        # 返回完整绘制后的图像（BGR 格式）和裁剪结果（RGB 格式）
        return image, crop_img

    def segment_on_cropped_region(self, image, kp_conf_threshold=0.25, seg_conf_threshold=0.4, line_thickness=2):
        """
        1. 使用 kpmodel 在输入图像上检测得到边界框，并根据边界框裁剪出目标区域。
        2. 在裁剪区域上调用分割模型进行分割，对预设类别（拼音对应）的 mask 查找轮廓并绘制。
        3. 返回绘制了分割轮廓的裁剪区域图像（BGR 格式）。

        参数：
           image: BGR格式的输入图像
           kp_conf_threshold: kpmodel的置信度阈值，默认0.25
           seg_conf_threshold: 分割模型置信度阈值，默认0.4
           line_thickness: 绘制轮廓的线宽

        返回：
           裁剪区域上绘制分割轮廓的图像（BGR格式）
        """
        # ---------------------------------------------
        # 第一步：使用 kpmodel 获取目标的边界框
        kp_results = self.kp_model(image, conf=kp_conf_threshold)
        bbox_coords = []
        for result in kp_results:
            if hasattr(result, 'boxes'):
                bboxes = result.boxes.data.cpu().numpy()
                for box in bboxes:
                    # 假定 box 格式为 [x_min, y_min, x_max, y_max, conf]
                    x_min, y_min, x_max, y_max, conf = box[:5]
                    bbox_coords.append((x_min, y_min, x_max, y_max))

        # 若检测到了边界框，则用最后一个框；否则使用整张图像
        if len(bbox_coords) > 0:
            x_min, y_min, x_max, y_max = bbox_coords[-1]
            # 注意边界框可能为浮点数，转换为整数
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        else:
            # 未检测到，则使用全图作为裁剪区域
            h, w = image.shape[:2]
            x_min, y_min, x_max, y_max = 0, 0, w, h

        # 裁剪目标区域
        cropped_img = image[y_min:y_max, x_min:x_max].copy()

        # ---------------------------------------------
        # 第二步：在裁剪区域上执行分割推理并绘制轮廓
        # 设定允许绘制轮廓的类别（拼音名称对应）
        allowed_categories = {
            "erchuiqu",  # 耳垂区
            "duierlunqu",  # 对耳轮区
            "duierpingqu",  # 对耳屏区
            "erzhouqu",  # 耳周区
            "erpingqu",  # 耳屏区
            "erjiaqu",  # 耳甲区
            "sanjiaowoqu" , # 三角窝区
            "erlunqu"
        }
        # 定义颜色映射
        color_map = {
            "erchuiqu": (255, 0, 0),
            "duierlunqu": (0, 255, 0),
            "duierpingqu": (0, 0, 255),
            "erzhouqu": (255, 255, 0),
            "erpingqu": (255, 0, 255),
            "erjiaqu": (0, 255, 255),
            "sanjiaowoqu": (128, 128, 0),
            "erlunqu": (0, 128, 128)
        }
        # 对裁剪区域进行分割推理
        seg_results = self.seg_model(cropped_img, conf=seg_conf_threshold, task="segment")
        if len(seg_results) == 0:
            return cropped_img  # 没有分割结果则直接返回裁剪的区域

        result = seg_results[0]
        if not hasattr(result, 'masks') or result.masks is None:
            return cropped_img

        # 分割模型返回的类别字典，假设形如 {class_id: class_name}，且名称为拼音
        classes = result.names
        masks = result.masks.data
        # 若模型返回了 boxes，则优先使用 boxes.cls 得到类别 id；否则默认 masks 序号对应
        if hasattr(result, 'boxes'):
            cls_array = result.boxes.cls.cpu().numpy()
        else:
            cls_array = list(range(len(masks)))

        # 获取裁剪区域尺寸
        crop_h, crop_w = cropped_img.shape[:2]
        for i in range(len(masks)):
            # 获取类别 id 及对应类别名称
            class_id = int(cls_array[i])
            class_name = classes[class_id]
            if class_name not in allowed_categories:
                continue

            # 取出 mask 并转换为 uint8 类型
            mask = masks[i].cpu().numpy().astype(np.uint8)
            mask_h, mask_w = mask.shape[:2]
            if (mask_h, mask_w) != (crop_h, crop_w):
                # 将 mask 调整至裁剪区域尺寸，使用 INTER_NEAREST 保持 mask 特性
                mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                color = color_map.get(class_name, (0, 255, 255))
                cv2.drawContours(cropped_img, [contour], -1, color, line_thickness)
                # 在轮廓附近标注类别名称，采用轮廓第一个点坐标
                if len(contour) > 0:
                    pt = tuple(contour[0][0])
                    cv2.putText(cropped_img, class_name, pt, cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, color, line_thickness, cv2.LINE_AA)
        return cropped_img




def main():
    # 请替换为实际的模型文件路径和字体文件路径
    seg_model_path = "weights/Auricular_acupoint_segment/best_side.pt"      # 分割模型文件路径
    kp_model_path = "weights/keypoint/er-pose-l.pt"        # 关键点模型文件路径
    font_path = "simhei.ttf"                    # 中文字体文件路径

    # 创建模型处理器实例
    processor = ModelProcessor(seg_model_path, kp_model_path, font_path)

    # 测试图像路径（确保该图像路径有效）
    # image_path = r"D:\Desktop\eeee\4.png"
    image_path = r"data/2025-07-11/yiyu/normal1/6/6_right_front.bmp"
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像，请检查路径:", image_path)
        return

    # 调用带有新增形状绘制功能的函数
    contours_with_colors, crop_img = processor.predict_and_visualize_frame_seg_and_selected_keypoints_with_region(
        image)

    # 保存结果图像（注意 crop_img 为RGB格式，这里转换为BGR后保存）
    output_path = "output_test.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
    print("处理结果已保存到：", output_path)

    full_image, crop_img = processor.draw_selected_segmentation_contours_and_crop(image.copy())
    cv2.imwrite("output_full.jpg", full_image)
    cv2.imwrite("output_crop.jpg", cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
    print("处理结果已保存！")

    segmentation_cropped = processor.segment_on_cropped_region(image.copy(),
                                                               kp_conf_threshold=0.25,
                                                               seg_conf_threshold=0.4,
                                                               line_thickness=2)
    cv2.imwrite("cropped_segmentation.jpg", segmentation_cropped)
    print("处理结果已保存！")

if __name__ == "__main__":
    main()



