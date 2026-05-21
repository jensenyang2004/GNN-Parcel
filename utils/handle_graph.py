import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_curve_from_image(image_path, color_name, x_max=20.0, y_max=1.0, num_points=21):
    """
    從裁切好的圖表圖片中提取特定顏色的曲線數據。
    
    參數:
        image_path: 圖片檔案路徑
        color_name: 目標顏色 ('blue', 'red', 'orange')
        x_max: 圖片最右側代表的 x 軸最大值
        y_max: 圖片最上方代表的 y 軸最大值
        num_points: 預期要取樣的資料點數量 (例如 0~20 每 1 單位取一點，共 21 點)
    """
    # 1. 讀取圖片並轉換為 HSV 色彩空間 (HSV 更適合用來過濾特定顏色)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"找不到圖片: {image_path}")

    height, width, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. 定義各顏色的 HSV 遮罩範圍
    # 注意：這些範圍可能需要根據截圖的實際色偏微調
    if color_name == 'blue':
        lower = np.array([100, 100, 50])
        upper = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif color_name == 'red':
        # 紅色在 HSV 色環的兩端 (0 附近和 180 附近)
        mask1 = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 100, 50]), np.array([180, 255, 255]))
        mask = mask1 | mask2
    elif color_name == 'orange':
        lower = np.array([11, 100, 50])
        upper = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    else:
        raise ValueError("顏色必須是 'blue', 'red' 或 'orange'")

    # 3. 將圖片在 X 軸方向切分成 num_points 個區塊，計算每個區塊內曲線的平均 Y 值
    bin_edges = np.linspace(0, width - 1, num_points + 1, dtype=int)
    
    extracted_x = []
    extracted_y = []

    for i in range(num_points):
        col_start = bin_edges[i]
        col_end = bin_edges[i+1]
        
        # 提取該垂直區間的遮罩
        slice_mask = mask[:, col_start:col_end]
        
        # 找出該區間內所有符合顏色的 y 座標 (圖片矩陣的列索引)
        y_coords, _ = np.where(slice_mask > 0)
        
        if len(y_coords) > 0:
            # 取中位數可以避免陰影區或雜訊造成的極端值影響
            median_y_pixel = np.median(y_coords)
            
            # 將像素座標轉換回圖表數值
            # 圖片的 Y 軸是往下遞增，但圖表是往上遞增，因此需要反轉 (height - y)
            real_y = ((height - median_y_pixel) / height) * y_max
            real_x = (i / (num_points - 1)) * x_max
            
            extracted_x.append(real_x)
            extracted_y.append(round(real_y, 4)) # 取小數點後四位

    return extracted_x, extracted_y

# ================= 執行區塊 =================
if __name__ == "__main__":
    # 替換成你的圖片檔名
    IMAGE_FILE = "./utils/graph.png"
    
    colors_to_extract = ['blue', 'red', 'orange']
    results = {}

    plt.figure(figsize=(8, 5))

    for color in colors_to_extract:
        try:
            x_data, y_data = extract_curve_from_image(IMAGE_FILE, color)
            results[color] = y_data
            
            # 畫出提取的結果來驗證
            plt.plot(x_data, y_data, marker='o', label=f'Extracted {color.capitalize()}')
            print(f"--- {color.capitalize()} Curve Data ---")
            print(f"Y values: {y_data}\n")
        except Exception as e:
            print(f"提取 {color} 失敗: {e}")

    # 設定圖表來驗證提取準確度
    plt.title("Extracted Curves Verification")
    plt.xlim(0, 20)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.show()