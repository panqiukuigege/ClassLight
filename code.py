import cv2
import numpy as np
import time
import os

class ClassroomOccupancyDetector:
    def __init__(self, config_path, weights_path, labels_path):
        """
        初始化教室占用检测器
        
        Args:
            config_path: YOLO配置文件路径
            weights_path: YOLO权重文件路径
            labels_path: 类别标签文件路径
        """
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.layer_names = self.net.getLayerNames()
        # 兼容 OpenCV 新旧版本（getUnconnectedOutLayers 返回格式不同）
        try:
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except TypeError:
            self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
        with open(labels_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # 灯光控制变量
        self.light_status = True  # True表示灯亮，False表示灯灭
        self.last_detection_time = time.time()
        self.inactivity_threshold = 30  # 30秒无检测到人则关闭灯

    def detect_objects(self, frame):
        """
        在帧中检测对象
        
        Args:
            frame: 输入图像帧
            
        Returns:
            boxes, confidences, class_ids, indexes
        """
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5 and self.classes[class_id] == "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        return boxes, confidences, class_ids, indexes

    def process_frame(self, frame):
        """
        处理单帧图像
        
        Args:
            frame: 输入图像帧
            
        Returns:
            processed_frame: 处理后的图像帧
            is_empty: 教室是否无人（True=无人，False=有人）
        """
        boxes, confidences, class_ids, indexes = self.detect_objects(frame)
        
        person_detected = False
        
        if len(indexes) > 0:
            # 兼容 NMSBoxes 返回格式（可能是列表或数组）
            indexes = indexes.flatten() if len(indexes.shape) > 1 else indexes
            for i in indexes:
                x, y, w, h = boxes[i]
                label = f"{self.classes[class_ids[i]]}: {confidences[i]:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                person_detected = True
        
        # 更新最后检测时间
        if person_detected:
            self.last_detection_time = time.time()
        
        # 判断是否需要开关灯
        current_time = time.time()
        if current_time - self.last_detection_time > self.inactivity_threshold:
            if self.light_status:
                self.light_status = False
                print("教室无人，自动关灯")
        else:
            if not self.light_status:
                self.light_status = True
                print("检测到人员，自动开灯")
        
        # 显示状态信息
        status_text = f"Light Status: {'ON' if self.light_status else 'OFF'}"
        occupancy_text = f"Occupancy: {'YES' if person_detected else 'NO'}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, occupancy_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, not person_detected

    def run_camera_detection(self, camera_index=0):
        """
        运行摄像头实时检测
        
        Args:
            camera_index: 摄像头索引，默认为0
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        print("开始检测... 按'q'键退出")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("无法接收帧")
                break
            
            processed_frame, is_empty = self.process_frame(frame)
            
            # 如果教室为空且灯亮着，输出提示
            if is_empty and self.light_status:
                print("教室没人 -> False")
            else:
                print("教室有人 -> True")
            
            cv2.imshow('Classroom Occupancy Detection', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    base = os.getcwd()  # 使用当前工作目录
    config_path = os.path.join(base, "yolov4-tiny.cfg")
    weights_path = os.path.join(base, "yolov4-tiny.weights")
    labels_path = os.path.join(base, "coco-tiny.names")  # 注意：建议用 coco.names，不是 coco-tiny.names

    # 检查文件是否存在
    for name, path in [("Config", config_path), ("Weights", weights_path), ("Labels", labels_path)]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} file missing: {path}")
    
    detector = ClassroomOccupancyDetector(config_path, weights_path, labels_path)
    detector.run_camera_detection()