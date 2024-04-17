import os
from ultralytics import YOLO
from PIL import Image

def process_image(image_path, output_dir):
    model = YOLO("C:\\Users\\User\\Pictures\\YoloV8\\preptn\\best.pt")
    results = model.predict(
        source=image_path,
        mode="predict",
        conf=0.1,  # 設置置信度閾值
        iou=0.45,  # IOU 閾值
        save=False,
        device="0"
    )

    # 取得圖檔名稱 (不包含副檔名)
    file_name = os.path.splitext(os.path.basename(image_path))[0]

    with open(os.path.join(output_dir, f'{file_name}.txt'), "w") as file:
        for result in results:
            # 對於每個檢測到的對象
            for box in result.boxes.xyxy:
                # 獲取坐標
                xmin, ymin, xmax, ymax = box.cpu().numpy()
                # 打開圖片
                image = Image.open(image_path)
                # 獲取圖片的寬度和高度
                image_w, image_h = image.size
                # 計算中心點的相對坐標
                x = (xmin + (xmax - xmin) / 2) / image_w
                y = (ymin + (ymax - ymin) / 2) / image_h
                # 計算寬度和高度的相對坐標
                w = (xmax - xmin) / image_w
                h = (ymax - ymin) / image_h

                # 將 x、y、w 和 h 寫入文件
                file.write(f'0 {x} {y} {w} {h}\n')

def main():
    input_dir = '.'  # 輸入圖檔所在的資料夾（當前資料夾）
    output_dir = '.'  # 輸出txt檔案的資料夾（當前資料夾）

    # 列舉資料夾內的所有PNG檔案
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            process_image(image_path, output_dir)

if __name__ == '__main__':
    main()
