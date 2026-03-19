import time

class FaceStateMachine:
    def __init__(self, hold_duration=1.5):
        """
        初始化状态机
        :param hold_duration: 触发动作所需保持状态的持续时间（秒），默认 1.5 秒
        """
        self.hold_duration = hold_duration
        self.current_state = None
        self.state_start_time = time.time()
        self.fired_for_state = False
        
        # 头部位置优先
        # 头部位置： left -> '[', right -> ']'
        # 情绪： happy -> '+', sad -> '-', angry -> '<', surprise -> '>', fear -> ',', disgust -> '.', neutral -> 'BACKSPACE'
        self.emotion_map = {
            'happy': '+',
            'sad': '-',
            'angry': '<',
            'surprise': '>',
            'fear': ',',
            'disgust': '.',
            'neutral': 'BACKSPACE'
        }

    def process(self, position, emotion, current_time=None):
        """
        处理当前识别到的头部位置和情绪
        :param position: 头部位置 ('left', 'center', 'right')
        :param emotion: 情绪字符串 ('happy', 'sad', 'surprise', 'angry', 'fear', 'disgust', 'neutral')
        :param current_time: 可选，用于测试的时间戳
        :return: 如果状态保持时间达到 hold_duration，则返回对应的 Brainfuck 字符(或 'BACKSPACE')，否则返回 None
        """
        if current_time is None:
            current_time = time.time()
            
        # 确定当前应当映射的字符
        char = None
        if position == 'left':
            char = '['
        elif position == 'right':
            char = ']'
        elif position == 'center':
            char = self.emotion_map.get(emotion)

        # 状态机逻辑
        if char != self.current_state:
            self.current_state = char
            self.state_start_time = current_time
            self.fired_for_state = False
            return None

        if char is None:
            return None

        if (not self.fired_for_state) and (current_time - self.state_start_time) >= self.hold_duration:
            self.fired_for_state = True
            return char

        return None

    def reset(self, current_time=None):
        if current_time is None:
            current_time = time.time()
        self.current_state = None
        self.state_start_time = current_time
        self.fired_for_state = False
