import collections
from typing import Optional, Tuple


class StateFilter:
    """
    滑动窗口多数投票滤波器，用于平滑识别到的状态（情绪和头部位置）。
    通过在时间窗口内取出现次数最多的状态，有效减少识别结果的抖动，平滑信号。
    """
    
    def __init__(
        self,
        window_size: int = 5,
        emotion_confirm_frames: int = 3,
        position_confirm_frames: int = 2,
        min_emotion_score: float = 0.0,
    ):
        """
        初始化状态滤波器。
        
        :param window_size: 滑动窗口的大小，即保留的历史帧数（默认 5）
        """
        if window_size < 1:
            raise ValueError("window_size 必须大于等于 1")
        if emotion_confirm_frames < 1 or position_confirm_frames < 1:
            raise ValueError("confirm_frames 必须大于等于 1")
            
        self.window_size = window_size
        self.emotion_confirm_frames = int(emotion_confirm_frames)
        self.position_confirm_frames = int(position_confirm_frames)
        self.min_emotion_score = float(min_emotion_score)
        self.emotions = collections.deque(maxlen=window_size)
        self.positions = collections.deque(maxlen=window_size)

        self._emotion_current = None
        self._emotion_candidate = None
        self._emotion_candidate_count = 0

        self._position_current = None
        self._position_candidate = None
        self._position_candidate_count = 0

    _UNSET = object()

    def update(self, emotion=_UNSET, position=_UNSET, emotion_score=None) -> Tuple[Optional[str], Optional[str]]:
        """
        将新一帧的识别结果加入滤波器，并返回平滑后的状态。
        
        :param emotion: 当前帧识别到的情绪 (str)
        :param position: 当前帧识别到的头部位置 (str)
        :param emotion_score: 当前帧主导情绪置信度（可选）
        :return: 平滑后的 (emotion, position) 元组
        """
        if emotion is not self._UNSET:
            if emotion is not None:
                if emotion_score is None or float(emotion_score) >= self.min_emotion_score:
                    self.emotions.append(emotion)
        if position is not self._UNSET:
            if position is not None:
                self.positions.append(position)
            
        vote_emotion = self._majority(self.emotions)
        vote_position = self._majority(self.positions)

        self._emotion_current, self._emotion_candidate, self._emotion_candidate_count = self._step_hysteresis(
            current=self._emotion_current,
            vote=vote_emotion,
            candidate=self._emotion_candidate,
            count=self._emotion_candidate_count,
            confirm_frames=self.emotion_confirm_frames,
        )
        self._position_current, self._position_candidate, self._position_candidate_count = self._step_hysteresis(
            current=self._position_current,
            vote=vote_position,
            candidate=self._position_candidate,
            count=self._position_candidate_count,
            confirm_frames=self.position_confirm_frames,
        )

        return self._emotion_current, self._position_current

    def _majority(self, dq) -> Optional[str]:
        if not dq:
            return None
        counts = collections.Counter(dq)
        return counts.most_common(1)[0][0]

    def _step_hysteresis(
        self,
        current: Optional[str],
        vote: Optional[str],
        candidate: Optional[str],
        count: int,
        confirm_frames: int,
    ):
        if vote is None:
            return current, None, 0
        if current is None:
            return vote, None, 0
        if vote == current:
            return current, None, 0
        if candidate == vote:
            count += 1
        else:
            candidate = vote
            count = 1
        if count >= confirm_frames:
            return candidate, None, 0
        return current, candidate, count

    def get_smoothed_state(self) -> Tuple[Optional[str], Optional[str]]:
        """
        计算并返回当前窗口内的多数投票结果。
        
        :return: 平滑后的 (emotion, position) 元组
        """
        return self._emotion_current, self._position_current

    def clear(self):
        """
        清空滑动窗口内的历史数据，通常在重置识别状态时调用。
        """
        self.emotions.clear()
        self.positions.clear()
        self._emotion_current = None
        self._emotion_candidate = None
        self._emotion_candidate_count = 0
        self._position_current = None
        self._position_candidate = None
        self._position_candidate_count = 0

if __name__ == "__main__":
    # 简单测试用例
    filter = StateFilter(window_size=3)
    
    # 模拟输入序列
    inputs = [
        ("happy", "center"),
        ("happy", "center"),
        ("sad", "left"),      # 噪点帧
        ("happy", "center"),
        ("sad", "left"),
        ("sad", "left"),
    ]
    
    print("Testing StateFilter...")
    for idx, (emo, pos) in enumerate(inputs):
        smoothed_emo, smoothed_pos = filter.update(emo, pos)
        print(f"Frame {idx+1}: Input=({emo}, {pos}) -> Smoothed=({smoothed_emo}, {smoothed_pos})")
