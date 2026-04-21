#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union, split


def cut_polygon_into_5_parts(polygon, names):
    """
    将输入的shapely Polygon沿最小外接矩形长边方向切割成5个部分。
    参数：
      polygon: shapely.geometry.Polygon，整块并集区域
      names: list，5个分段的类别名称，按照“从上到下”的顺序排列
    返回：包含5个字典，每个字典含有：
           'contour': np.ndarray格式轮廓
           'class_name': 分段名称
    若切割后多边形数量不等于5，则返回整体轮廓，类别为'erzhouqu'
    """
    # 计算最小外接矩形
    rect = polygon.minimum_rotated_rectangle
    box_coords = np.array(rect.exterior.coords[:-1])  # 去除闭合重复顶点

    # 寻找长边
    edge_lengths = [np.linalg.norm(box_coords[i] - box_coords[(i + 1) % 4]) for i in range(4)]
    long_edge_idx = np.argmax(edge_lengths)
    long_edge_start = box_coords[long_edge_idx]
    long_edge_end = box_coords[(long_edge_idx + 1) % 4]
    long_vec = long_edge_end - long_edge_start
    long_len = np.linalg.norm(long_vec)
    long_dir = long_vec / long_len

    # 计算垂直于长边方向（短边方向）
    short_edge_idx = (long_edge_idx + 1) % 4
    short_vec = box_coords[short_edge_idx] - box_coords[(short_edge_idx + 1) % 4]
    short_vec = -short_vec  # 调整方向
    short_len = np.linalg.norm(short_vec)
    short_dir = short_vec / short_len

    # 生成4条切割线，将长边分成5份
    segment_length = long_len / 5
    cut_lines = []
    for i in range(1, 5):
        base_point = long_edge_start + long_dir * segment_length * i
        p1 = base_point
        p2 = base_point + short_dir * short_len * 1.5  # 多延展以确保完整切割
        cut_lines.append(LineString([tuple(p1), tuple(p2)]))

    # 迭代切割得到若干子多边形
    polygons = [polygon]
    for cut_line in cut_lines:
        new_polygons = []
        for poly in polygons:
            splitted = split_polygon_by_line(poly, cut_line)
            new_polygons.extend(splitted)
        polygons = new_polygons

    # 过滤非Polygon类型
    polygons = [p for p in polygons if p.geom_type == 'Polygon']

    if len(polygons) != 5:
        print(f"警告: 切割后多边形数量不为5，而是{len(polygons)}，返回整体轮廓。")
        contour_np = np.array(polygon.exterior.coords[:-1], dtype=np.int32)
        return [{'contour': contour_np, 'class_name': 'erzhouqu'}]

    # 根据质心纵坐标（y）升序排列，实现“从上到下”
    polygons_sorted = sorted(polygons, key=lambda p: p.centroid.y)

    result = []
    for poly, name in zip(polygons_sorted, names):
        coords = np.array(poly.exterior.coords[:-1], dtype=np.int32)
        result.append({'contour': coords, 'class_name': name})
    return result


def split_polygon_by_line(polygon, line):
    """
    根据一条直线切分输入的polygon，返回切割后的多边形列表（只保留Polygon类型）。
    """
    splitted = split(polygon, line)
    return [geom for geom in splitted.geoms if geom.geom_type == 'Polygon']


def output_vessel_diagnosis(img, mask, output_dir, contours_with_colors,r_image2):
    """
    根据输入原图img和分割遮罩mask，利用model对血管区域进行诊断、分割并绘图。
    输入：
      img: BGR格式的原图（numpy数组）
      mask: 分割结果图像（一般为mask）——要求与img尺寸一致
      output_dir: 指定输出目录（内部会保存多张调试图像）
      contours_with_colors: 一个列表，每个元素为{'class_name': 类别, 'contour': 轮廓, 'color': 颜色}；注意'erzhouqu'区域会后续切分成5部分
    该函数执行以下操作：
      1. 首先根据 contours_with_colors 获取其外接矩形，并将原图和mask裁剪到该矩形内部，
         同时调整所有轮廓的坐标，使得后续操作在裁剪后的区域内进行，输出图像大小即为此矩形大小。
      2. 对裁剪后的图像进行颜色转换和阈值处理，保存原图和处理后的mask图像。
      3. 针对'erzhouqu'区域进行多边形并集和切割（切分成5部分）。
      4. 遍历所有区域，对各个区域进行白色像素计数（检测区域内是否出现血丝），并绘制检测结果。
    返回值：
      mask_bgr: 绘制了检测标注的mask图（彩色BGR图像）
      result: 一个列表，每个元素为{'class_name': 类别, 'flag': 0/1}，flag=1表示当前区域检测到血丝
    """

    # 1. 根据所有 contours_with_colors 计算外接矩形，然后对图像进行裁剪
    all_points = []
    for item in contours_with_colors:
        # 保证轮廓为二维数组
        pts = item['contour'].reshape(-1, 2)
        all_points.append(pts)
    if all_points:
        all_points = np.concatenate(all_points, axis=0)
        # 计算外接矩形
        x, y, w, h = cv2.boundingRect(all_points.astype(np.int32))
        # 将负值抹零
        x = max(x, 0)
        y = max(y, 0)
        print(x,y,w,h)
        # 裁剪原图和mask
        img = img[y:y+h, x:x+w]
        r_image2 = r_image2.crop((x, y, x + w, y + h)).resize((2160, 3840))
        mask = mask[y:y+h, x:x+w]
        # 修改 contours_with_colors 中每个轮廓的坐标，减去裁剪的左上角偏移量
        for item in contours_with_colors:
            item['contour'] = item['contour'] - np.array([[[x, y]]])
    else:
        # 如果contours_with_colors为空，则不做裁剪
        print("没有有效的轮廓，跳过裁剪。")

    # 如果输出目录不存在，创建之
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 2. 图像预处理及保存调试图
    # 注意：这里先将img从BGR转为RGB（后续处理和存储的图片均为裁剪后的区域）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(output_dir, 'origin_img.jpg'), img)

    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, 'mask1.jpg'), mask_gray)
    _, mask_bin = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(output_dir, 'mask.jpg'), mask_bin)

    mask_bgr = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
    img_copy = img.copy()

    # 3. 处理'erzhouqu'区域（后续按要求切分成5部分）
    # 定义erzhouqu区域切割后分段的名称（顺序为：上到下）
    split_names = ['zhi2', 'wan', 'zhou', 'jian', 'suogu']

    erzhouqu_polys = []
    erzhouqu_color = None
    other_contours = []  # 非erzhouqu区域

    for item in contours_with_colors:
        if item['class_name'] == 'erzhouqu':
            contour = item['contour']
            erzhouqu_color = item['color']
            if len(contour) < 4:
                continue
            poly = Polygon(contour.reshape(-1, 2))
            if not poly.is_valid:
                poly = poly.buffer(0)  # 尝试修复
            erzhouqu_polys.append(poly)
        else:
            other_contours.append(item)

    # 对所有erzhouqu轮廓做并集
    if erzhouqu_polys:
        union_poly = unary_union(erzhouqu_polys)
        if union_poly.geom_type == 'MultiPolygon':
            union_poly = max(union_poly.geoms, key=lambda p: p.area)
        splitted_erzhouqu = cut_polygon_into_5_parts(union_poly, split_names)
        for part in splitted_erzhouqu:
            part['color'] = erzhouqu_color
    else:
        splitted_erzhouqu = []

    new_contours = other_contours + splitted_erzhouqu

    # 4. 遍历区域，统计白色像素数量，并绘制结果
    result = []
    for item in new_contours:
        contour = item['contour']
        itname = item['class_name']
        color = item.get('color', (0, 255, 0))

        contour_mask = np.zeros(mask_bin.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)

        masked_region = cv2.bitwise_and(mask_bin, contour_mask)
        white_pixel_count = cv2.countNonZero(masked_region)
        # 输出每个区域的白色像素数量
        print(f'{itname} has {white_pixel_count} white_pixel')
        flag = 1 if white_pixel_count > 0 else 0

        result.append({'class_name': itname, 'flag': flag})
        cv2.drawContours(img_copy, [contour], -1, color, 2)
        result_color = (0, 255, 0) if flag == 0 else (0, 0, 255)
        cv2.drawContours(mask_bgr, [contour], -1, result_color, 2)

        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(mask_bgr,
                        f"{itname}:{flag}",
                        (cx - 10, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.8,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA)

    cv2.imwrite(os.path.join(output_dir, 'segment_result.jpg'), img_copy)
    cv2.imwrite(os.path.join(output_dir, 'mask_annotation.jpg'), mask_bgr)

    # 返回mask标注图和result（后续调用者可根据result构造输出字符串）
    return mask_bgr, result,r_image2

def get_mapped_name(raw_name, mapping_path='mapping.txt'):
    """
    根据输入的原始名称，从 mapping.txt 中加载映射关系，
    返回对应的中文名称。如果未找到映射，则返回 raw_name 本身。

    参数:
      raw_name: 字符串，原始名称，例如 'sanjiaowoqu'
      mapping_path: 映射文件路径，默认 'mapping.txt'
                   文件内容要求每行格式为 "key:中文"，例如 "sanjiaowoqu:三角窝区"
    返回:
      对应的映射中文名称，如 '三角窝区'；未找到则返回 raw_name
    """
    if not os.path.exists(mapping_path):
        # 映射文件不存在直接返回原始名称
        return raw_name

    mapping = {}
    with open(mapping_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, value = line.split(':', 1)
            mapping[key.strip()] = value.strip()

    return mapping.get(raw_name, raw_name)


def vessel_diagnosis(image, output_dir, unet_model, img_name,contours_with_colors):
    """
    vessel_diagnosis: 对单张图像进行血管检测诊断
    输入：
      image: PIL 图像对象（待检测图像）
      output_dir: 用于保存中间及最终结果的输出目录（不存在则自动创建）
      unet_model: 分割模型对象（需具备detect_image2方法），例如Unet实例
      model_processor: 诊断模型对象（例如ModelProcessor_front实例），需具备predict_and_visualize_frame方法
      img_name: 图像文件名（用于保存时命名）
      area_threshold: 连通域面积阈值，低于该面积的连通域将被去除
    内部流程：
      1. 利用 unet_model.detect_image2 获得分割结果图（mask）；
      2. 将原图转换为 numpy 格式，并按要求做颜色通道处理；
      3. 对 num_mask 进行小连通域去除处理；
      4. 调用 output_vessel_diagnosis 完成轮廓提取、检测及保存图像；
      5. 根据检测结果统计出现血丝的区域，并构造输出字符串 output_str；
    返回值：
      mask_bgr: 保存于 output_dir/mask_annotation.jpg 中的标注图（BGR格式的numpy数组）
      output_str: 文字描述，若检测到血丝，则输出“xx, xx出现血丝”，否则返回“没有出现血丝的穴位”
      output_rgb: 处理后的结果图（RGB格式的numpy数组）
    """

    # 构建输出目录（使用 img_name 作为子文件夹名）
    output_dir = os.path.join(output_dir, img_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 调用分割模型，得到两个输出（r_image为mask图，r_image2为彩色预测图）
    r_image, r_image2 = unet_model.detect_image2(image)

    # 将原图转换为 numpy 数组，并转换颜色（若原图为RGB，而opencv要求BGR）
    num_img = np.asarray(image)
    num_img = cv2.cvtColor(num_img, cv2.COLOR_RGB2BGR)

    # 将分割模型得到的mask转换为 numpy 数组
    num_mask = np.asarray(r_image)


    # 调用血管诊断函数，返回 mask 标注图以及检测结果列表
    mask_bgr, result,r_image2 = output_vessel_diagnosis(num_img, num_mask,  output_dir,contours_with_colors,r_image2)

    # 根据检测结果统计出现血丝的区域
    names_with_flag_1 = [get_mapped_name(item['class_name']) for item in result if item['flag'] == 1]
    if names_with_flag_1:
        output_str = ",".join(names_with_flag_1) + "出现血络"
    else:
        output_str = "没有出现血络的穴位"

    # 可选：将 output_str 保存到文本文件
    txt_path = os.path.join(output_dir, "vessel_diagnosis_result.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(output_str)

    # 将 PIL 图像转换为 NumPy 数组
    r_image2_np = np.array(r_image2)
    # 反转颜色通道顺序，实现 BGR -> RGB（若原图为RGB则可忽略此步骤）
    r_image2_np = r_image2_np[..., ::-1]
    # 利用转换后的数组创建新的 PIL 图像，并指定模式为 RGB
    r_image2_final = Image.fromarray(r_image2_np, "RGB")

    # 保存图像
    r_image.save(os.path.join(output_dir, img_name + '.jpg'))
    r_image2_final.save(os.path.join(output_dir, img_name + '_new.jpg'))

    output_rgb = np.asarray(r_image2_final)

    return mask_bgr, output_str, output_rgb


# ------------------------------------------------------------------------------
# 以下为示例调用代码，当此文件作为模块引用时，可直接调用vessel_diagnosis函数，
# 而不必执行下面的测试代码。
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # 示例：初始化模型及参数
    # 请根据实际情况修改模型加载方式和模型权重路径
    from unet import Unet  # 假设unet.py中定义了Unet
    from unet_utils.auricular_segmentation import ModelProcessor_front

    # 初始化模型（保证模型文件路径正确）
    seg_model_path = r'best_side.pt'
    kp_model_path = r'best-pose.pt'
    model_processor = ModelProcessor_front(seg_model_path, kp_model_path)
    unet_model = Unet()  # 或者加载预训练模型

    # 指定待检测图像和输出目录
    img_name = '2_left_front'
    test_image_path = f"img/{img_name}.jpg"  # 待检测图片路径
    output_directory = f"output/img_out_{int(time.time())}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    try:
        test_image = Image.open(test_image_path)
    except Exception as e:
        raise ValueError(f"打开图像失败：{e}")

    # 调用检测诊断函数
    mask_annotation, diagnosis_str = vessel_diagnosis(test_image, output_directory, unet_model, model_processor,img_name)
    print("诊断结果：", diagnosis_str)
    # 此外，生成的图像文件均保存在output_directory目录下。