from DexiNed_master.dexined_predict import predict_single_image
import cv2
import numpy as np
TEST_IMG_WIDTH  = 512
TEST_IMG_HEIGHT = 512
test_img_path = "/dataset/zhuluoji/DexiNed-master/DexiNed-master/result/BIPED2CLASSIC/avg/1748657341.151986.jpg"
single_img = cv2.imread(test_img_path)
if single_img is None:
    raise ValueError("加载图像失败，请检查路径！")
single_img = cv2.resize(single_img, (TEST_IMG_WIDTH, TEST_IMG_HEIGHT))
single_img = single_img.astype(np.float32)
single_img = np.transpose(single_img, (2, 0, 1))
prediction_tensor = predict_single_image(single_img)
print("预测tensor结果：", prediction_tensor)

prediction_np = prediction_tensor.cpu().numpy()

# 将预测结果数值范围 [0,1] 转换为 [0,255]，并转换为 uint8 类型
result_uint8 = (prediction_np * 255).clip(0, 255).astype(np.uint8)

# 保存图片到磁盘，使用 cv2.imwrite 保存为灰度图
save_path = "predicted_result.png"
cv2.imwrite(save_path, result_uint8)
print(f"预测结果已保存至 {save_path}")

