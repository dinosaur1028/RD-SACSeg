#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image
from utils.auricular_segmentation import ModelProcessor_front

from unet import Unet_ONNX, Unet
import os
os.chdir('/dataset/zhuluoji/unet/unet-pytorch-main')
seg_model_path = r'./model_data/models/best_side.pt'
kp_model_path = r'./model_data/models/best-pose.pt'
model = ModelProcessor_front(seg_model_path,kp_model_path)

from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union, split
import numpy as np
import cv2
import os
name_map = {
    'erzhouqu': '耳周区',
    'zhi2': '指',
    'wan': '腕',
    'zhou': '肘',
    'jian': '肩',
    'suogu': '锁骨',
    'kou': '口',
    'shidao': '食道',
    'benmen': '贲门',
    'wei': '胃',
    'shierzhichang': '十二指肠',
    'xiaochang': '小肠',
    'dachang': '大肠',
    'tingjiao': '庭角',
    'pangguang': '膀胱',
    'shen': '肾',
    'yidan': '胰胆',
    'gan': '肝',
    'pi': '脾',
    'fei': '肺',
    'xin': '心',
    'qiguan': '气管',
    'sanjiao': '三焦',
    'neifenmi': '内分泌',
    'gen': '跟',
    'zhi': '趾',
    'huai': '踝',
    'xi': '膝',
    'kuan': '髋',
    'zuogushenjing': '坐骨神经',
    'tun': '臀',
    'fu': '腹',
    'yaodizhui': '腰骶椎',
    'xiong': '胸',
    'xiongzhui': '胸椎',
    'jing': '颈',
    'jingzhui': '颈椎',
    
}

def cut_polygon_into_5_parts(polygon, names):
    """
    polygon: shapely Polygon，整体并集区域
    names: 5个分段名字列表
    返回 [{'contour': np.ndarray, 'class_name': str}, ...] 五个子轮廓字典列表
    """
    # 计算最小外接矩形
    rect = polygon.minimum_rotated_rectangle
    box_coords = np.array(rect.exterior.coords[:-1])  # 4点，顺序闭合，去除最后重复点

    # 找长边
    edge_lengths = [np.linalg.norm(box_coords[i] - box_coords[(i+1)%4]) for i in range(4)]
    long_edge_idx = np.argmax(edge_lengths)
    long_edge_start = box_coords[long_edge_idx]
    long_edge_end = box_coords[(long_edge_idx+1)%4]
    long_vec = long_edge_end - long_edge_start
    long_len = np.linalg.norm(long_vec)
    long_dir = long_vec / long_len

    # 边垂直方向（短边）
    short_edge_idx = (long_edge_idx + 1) % 4
    short_vec = box_coords[short_edge_idx] - box_coords[(short_edge_idx+1)%4]
    short_vec = -short_vec  # 调整方向一致
    short_len = np.linalg.norm(short_vec)
    short_dir = short_vec / short_len

    # 计算4条切割线（将长边分成5份）
    segment_length = long_len / 5
    cut_lines = []
    for i in range(1, 5):
        base_point = long_edge_start + long_dir * segment_length * i
        p1 = base_point
        p2 = base_point + short_dir * short_len * 1.5  # 多延展一点，确保完全切割
        cut_lines.append(LineString([tuple(p1), tuple(p2)]))

    # 迭代切割
    polygons = [polygon]
    for cut_line in cut_lines:
        new_polygons = []
        for poly in polygons:
            splitted = split_polygon_by_line(poly, cut_line)
            new_polygons.extend(splitted)
        polygons = new_polygons

    # 过滤多边形，确保只有Polygon
    polygons = [p for p in polygons if p.geom_type == 'Polygon']

    if len(polygons) != 5:
        print(f"警告: 切割后多边形数量不为5，而是{len(polygons)}，返回整体轮廓。")
        contour_np = np.array(polygon.exterior.coords[:-1], dtype=np.int32)
        return [{'contour': contour_np, 'class_name': 'erzhouqu'}]

    # 按多边形质心的纵坐标（y）升序排序，实现“从上到下”
    polygons_sorted = sorted(polygons, key=lambda p: p.centroid.y)

    result = []
    for poly, name in zip(polygons_sorted, names):
        coords = np.array(poly.exterior.coords[:-1], dtype=np.int32)
        result.append({'contour': coords, 'class_name': name})

    return result

def split_polygon_by_line(polygon, line):
    """
    按line切割polygon，返回多边形列表，只保留Polygon类型
    """
    splitted = split(polygon, line)
    return [geom for geom in splitted.geoms if geom.geom_type == 'Polygon']

def output_vessel_diagnosis(img, mask, model, output_dir):
    import shapely  # 确保导入

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    contours_with_colors, frame = model.predict_and_visualize_frame(img)
    cv2.imwrite(f'{output_dir}/origin_img.jpg', img)

    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'{output_dir}/mask1.jpg', mask_gray)
    _, mask_bin = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f'{output_dir}/mask.jpg', mask_bin)

    mask_bgr = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
    img_copy = img.copy()

    split_names = ['zhi2', 'wan', 'zhou', 'jian', 'suogu']

    # 先收集erzhouqu对应的多边形，做并集
    erzhouqu_polys = []
    erzhouqu_color = None

    other_contours = []  # 其他类别原样存储

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

    # 并集
    if erzhouqu_polys:
        union_poly = unary_union(erzhouqu_polys)
        # 如果是MultiPolygon，合并成一个Polygon（或保持MultiPolygon）
        if union_poly.geom_type == 'MultiPolygon':
            # 这里你可以选择怎么处理，比如取最大面积的一个Polygon
            union_poly = max(union_poly.geoms, key=lambda p: p.area)
        # 切割并返回5个子轮廓
        splitted_erzhouqu = cut_polygon_into_5_parts(union_poly, split_names)
        # 给分割轮廓加颜色
        for part in splitted_erzhouqu:
            part['color'] = erzhouqu_color
    else:
        splitted_erzhouqu = []

    # 构造新的轮廓列表：剔除原erzhouqu，加入新的5个分段
    new_contours = other_contours + splitted_erzhouqu

    result = []

    for item in new_contours:
        contour = item['contour']
        itname = item['class_name']
        color = item.get('color', (0, 255, 0))

        contour_mask = np.zeros(mask_bin.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)

        masked_region = cv2.bitwise_and(mask_bin, contour_mask)
        white_pixel_count = cv2.countNonZero(masked_region)
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

    cv2.imwrite(f'{output_dir}/segment_result.jpg', img_copy)
    cv2.imwrite(f'{output_dir}/mask_annotation.jpg', mask_bgr)

    return result





if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    #-------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #   'predict_onnx'      表示利用导出的onnx模型进行预测，相关参数的修改在unet.py_346行左右处的Unet_ONNX
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    #-------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    #
    #   count、name_classes仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    count           = False
    # name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # name_classes    = ["background","cat","dog"]
    name_classes    = ["background","vessel"]
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img0509/"

    dir_save_path   = f"output/img_out_{int(time.time())}/"
    if not os.path.exists(dir_save_path):
        os.mkdir(dir_save_path)

    #-------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode != "predict_onnx":
        unet = Unet()
    else:
        yolo = Unet_ONNX()

    if mode == "predict":
        '''
        predict.py有几个注意点
        1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
        具体流程可以参考get_miou_prediction.py，在get_miou_prediction.py即实现了遍历。
        2、如果想要保存，利用r_image.save("img.jpg")即可保存。
        3、如果想要原图和分割图不混合，可以把blend参数设置成False。
        4、如果想根据mask获取对应的区域，可以参考detect_image函数中，利用预测结果绘图的部分，判断每一个像素点的种类，然后根据种类获取对应的部分。
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = unet.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(unet.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = unet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
        
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image,r_image2     = unet.detect_image2(image)
                num_img = np.asarray(image)
                num_img = cv2.cvtColor(num_img,cv2.COLOR_BGR2RGB)
                num_mask = np.asarray(r_image)
                save_path = os.path.join(dir_save_path, img_name.split('.')[0])
                result = output_vessel_diagnosis(num_img,num_mask,model,save_path)
                # 筛选flag=1的class_name
                names_with_flag_1 = [name_map.get(item['class_name'], item['class_name']) for item in result if item['flag'] == 1]
                txt_path = f"{save_path}/classes.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    for item in result:
                        f.write(item['class_name'])

                if names_with_flag_1:
                    output_str = ",".join(names_with_flag_1) + "出现血丝"
                else:
                    output_str = "没有出现血丝的血管"

                # 保存到txt文件
                txt_path = f"{save_path}/vessel_diagnosis_result.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(output_str)
                # if not os.path.exists(dir_save_path):
                #     os.makedirs(dir_save_path)
                r_image.save(os.path.join(save_path, img_name))
                r_image2.save(os.path.join(save_path, img_name.split('.')[0]+'_new.jpg'))
    elif mode == "export_onnx":
        unet.convert_to_onnx(simplify, onnx_save_path)
                
    elif mode == "predict_onnx":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
