import cv2
import numpy as np
from deepface import DeepFace


def _is_fallback_full_frame(region: dict, frame_w: int, frame_h: int) -> bool:
    x = int(region.get("x", -1) or -1)
    y = int(region.get("y", -1) or -1)
    w = int(region.get("w", -1) or -1)
    h = int(region.get("h", -1) or -1)
    return x <= 1 and y <= 1 and w >= (frame_w - 2) and h >= (frame_h - 2)


def _inter_area(ax, ay, aw, ah, bx, by, bw, bh):
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    iw = x2 - x1
    ih = y2 - y1
    if iw <= 0 or ih <= 0:
        return 0.0
    return float(iw * ih)


def classify_horizontal_position_by_area(region: dict, frame_width: int, frame_height: int):
    x = float(region.get("x", 0))
    y = float(region.get("y", 0))
    w = float(region.get("w", 0))
    h = float(region.get("h", 0))
    if w <= 0 or h <= 0 or frame_width <= 0 or frame_height <= 0:
        return None, {}

    bbox_area = w * h
    third_w = float(frame_width) / 3.0

    left_area = _inter_area(x, y, w, h, 0.0, 0.0, third_w, float(frame_height))
    right_area = _inter_area(x, y, w, h, 2.0 * third_w, 0.0, third_w, float(frame_height))

    ratios = {
        "left": left_area / bbox_area,
        "right": right_area / bbox_area,
        "center": max(0.0, 1.0 - (left_area + right_area) / bbox_area),
    }

    if ratios["left"] >= 0.5:
        return "left", ratios
    if ratios["right"] >= 0.5:
        return "right", ratios
    return "center", ratios


def classify_vertical_position_by_area(region: dict, frame_width: int, frame_height: int):
    x = float(region.get("x", 0))
    y = float(region.get("y", 0))
    w = float(region.get("w", 0))
    h = float(region.get("h", 0))
    if w <= 0 or h <= 0 or frame_width <= 0 or frame_height <= 0:
        return None, {}

    bbox_area = w * h
    third_h = float(frame_height) / 3.0

    top_area = _inter_area(x, y, w, h, 0.0, 0.0, float(frame_width), third_h)
    bottom_area = _inter_area(x, y, w, h, 0.0, 2.0 * third_h, float(frame_width), third_h)

    ratios = {
        "up": top_area / bbox_area,
        "down": bottom_area / bbox_area,
        "middle": max(0.0, 1.0 - (top_area + bottom_area) / bbox_area),
    }

    if ratios["up"] >= 0.5:
        return "up", ratios
    if ratios["down"] >= 0.5:
        return "down", ratios
    return "middle", ratios


class FaceRecognizer:
    def __init__(
        self,
        enforce_detection=False,
        min_emotion_confidence=40.0,
        min_neutral_confidence=70.0,
        position_deadzone=0.08,
    ):
        """
        初始化人脸情绪和位置识别器
        :param enforce_detection: 如果为True，当未检测到人脸时会抛出异常；如果为False，会返回空结果。
        """
        self.enforce_detection = enforce_detection
        self.min_emotion_confidence = float(min_emotion_confidence)
        self.min_neutral_confidence = float(min_neutral_confidence)
        self.position_deadzone = float(position_deadzone)

    def analyze_frame(self, frame: np.ndarray) -> list:
        """
        分析输入图像帧，获取检测到的人脸的情绪(7种)及头部位置(左/右/中)
        
        :param frame: cv2 读取的 numpy array 图像帧 (BGR)
        :return: 包含每个检测到的人脸信息的列表，格式为:
                 [{
                     'emotions': dict,           # 7种情绪分数
                     'dominant_emotion': str,    # 主导情绪
                     'position': str,            # 头部位置 ('left', 'center', 'right')
                     'box': dict                 # 边界框信息 {'x': int, 'y': int, 'w': int, 'h': int}
                 }, ...]
        """
        results = []
        try:
            height, width = frame.shape[:2]
            third_width = width / 3

            # 使用 DeepFace 进行情绪分析
            # actions=['emotion'] 将提取7种基础情绪：angry, disgust, fear, happy, sad, surprise, neutral
            analyses = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=self.enforce_detection,
                detector_backend='opencv',
                silent=True
            )

            # DeepFace 返回结果可能是单个字典或字典列表
            if not isinstance(analyses, list):
                analyses = [analyses]

            for face in analyses:
                # 获取7种情绪数据和主导情绪
                emotions = face.get('emotion', {})
                dominant_emotion = None
                dominant_score = 0.0
                if emotions:
                    dominant_emotion = max(emotions, key=emotions.get)
                    dominant_score = float(emotions.get(dominant_emotion, 0.0))

                    if dominant_score < self.min_emotion_confidence:
                        dominant_emotion = None
                    elif dominant_emotion == 'neutral' and dominant_score < self.min_neutral_confidence:
                        dominant_emotion = None

                # 获取边界框数据
                region = face.get('region', {})
                if _is_fallback_full_frame(region, width, height):
                    continue
                x = region.get('x', 0)
                y = region.get('y', 0)
                w = region.get('w', 0)
                h = region.get('h', 0)

                # 判断头部位置
                position, horizontal_ratios = classify_horizontal_position_by_area(region, width, height)
                vertical_position, vertical_ratios = classify_vertical_position_by_area(region, width, height)

                results.append({
                    'emotions': emotions,
                    'dominant_emotion': dominant_emotion,
                    'dominant_score': dominant_score,
                    'position': position,
                    'position_ratios': horizontal_ratios,
                    'vertical_position': vertical_position,
                    'vertical_ratios': vertical_ratios,
                    'box': region
                })

        except ValueError as e:
            # DeepFace 在 enforce_detection=True 且未检测到人脸时会抛出 ValueError
            print(f"Face detection warning: {e}")
        except Exception as e:
            print(f"Error during face analysis: {e}")

        return results

if __name__ == "__main__":
    # 简单的测试代码
    print("Testing FaceRecognizer...")
    # 创建一个纯黑色测试图像，尺寸为 640x480 (宽x高)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 实例化识别器，关闭 enforce_detection 以防纯黑图像报错停止
    recognizer = FaceRecognizer(enforce_detection=False)
    
    print("Analyzing test frame (black image, expecting no faces or random behavior based on backend)...")
    res = recognizer.analyze_frame(test_frame)
    print("Analysis Result:")
    for i, face_info in enumerate(res):
        print(f"Face {i+1}:")
        print(f"  - Position: {face_info['position']}")
        print(f"  - Dominant Emotion: {face_info['dominant_emotion']}")
        print(f"  - Emotions Details: {face_info['emotions']}")
        print(f"  - Bounding Box: {face_info['box']}")
    print("Test finished.")
