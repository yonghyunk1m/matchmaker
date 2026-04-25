# # 파일 경로: matchmaker/matchmaker/performance_plan.py
# import numpy as np
# import os
# import pickle

# class PerformancePlan:
#     def __init__(self, score_len_beats, save_path="vivace_memory.pkl"):
#         """
#         VIVACE Performance Plan (논문 Section IV)
#         :param score_len_beats: 악보의 총 길이 (비트 단위)
#         :param save_path: 기억을 저장할 파일 경로
#         """
#         # 데이터 구조: [비트 인덱스] -> 값
#         # 1. Tempo Ratio Plan (1.0 = 정템포, >1.0 = 빠름, <1.0 = 느림)
#         self.tempo_plan = np.ones(int(score_len_beats) + 10) 
        
#         # 2. Dynamics Plan (0~127 MIDI Velocity)
#         self.dynamics_plan = np.full(int(score_len_beats) + 10, 64.0)
        
#         # 3. Visit Count (해당 구간을 몇 번 연습했는지)
#         self.visit_counts = np.zeros(int(score_len_beats) + 10)
        
#         self.save_path = save_path
#         self.load_plan() # 파일이 있으면 불러오기

#     def update(self, beat, input_tempo_ratio, input_velocity, confidence):
#         """
#         [학습 모드] 논문 수식 (5): Confidence-Gated Update
#         신뢰도가 높을 때만 학습하여 에러 누적 방지
#         """
#         idx = int(beat)
#         if idx < 0 or idx >= len(self.tempo_plan): return

#         # 학습률(Alpha) 계산: 기본 0.2 * 신뢰도 (논문 Section IV-A)
#         # 신뢰도가 낮으면 alpha -> 0이 되어 학습하지 않음
#         alpha = 0.2 * confidence

#         # 1. 템포 업데이트 (WEMA: Weighted Exponential Moving Average)
#         # P_new = (1-a)*P_old + a*Input
#         self.tempo_plan[idx] = (1 - alpha) * self.tempo_plan[idx] + alpha * input_tempo_ratio
        
#         # 2. 다이내믹 업데이트
#         if input_velocity > 0: # 유효한 볼륨일 때만
#             self.dynamics_plan[idx] = (1 - alpha) * self.dynamics_plan[idx] + alpha * input_velocity
            
#         self.visit_counts[idx] += 1

#     def get_params(self, beat):
#         """
#         [연주 모드] 현재 위치의 학습된 파라미터 반환
#         """
#         idx = int(beat)
#         if idx < 0 or idx >= len(self.tempo_plan): return 1.0, 64
#         return self.tempo_plan[idx], self.dynamics_plan[idx]

#     def save_plan(self):
#         with open(self.save_path, 'wb') as f:
#             pickle.dump((self.tempo_plan, self.dynamics_plan), f)
#         print(f"[VIVACE] Performance Plan saved to {self.save_path}")

#     def load_plan(self):
#         if os.path.exists(self.save_path):
#             try:
#                 with open(self.save_path, 'rb') as f:
#                     self.tempo_plan, self.dynamics_plan = pickle.load(f)
#                 print(f"[VIVACE] Loaded existing memory from {self.save_path}")
#             except:
#                 print("[VIVACE] Failed to load memory, starting fresh.")

# New (Feb 05)
import numpy as np
import os
import pickle

class PerformancePlan:
    def __init__(self, score_len_beats, save_path="vivace_memory.pkl"):
        """
        VIVACE Performance Plan
        :param score_len_beats: Length of score in beats
        :param save_path: Path to memory file
        """
        self.score_len_beats = int(score_len_beats) + 10
        
        # 1. Plans (Tempo Ratio, Dynamics)
        self.tempo_plan = np.ones(self.score_len_beats)
        self.dynamics_plan = np.full(self.score_len_beats, 64.0)
        self.visit_counts = np.zeros(self.score_len_beats)
        self.locked_beats = np.zeros(self.score_len_beats, dtype=bool)
        # [SCORPION] Adaptive window: per-beat window size (seconds).
        # None = use default. Set by update_window_from_errors after rehearsal.
        self.window_plan = np.zeros(self.score_len_beats)  # 0 = use default

        self.save_path = save_path
        self.is_locked = False # [VIVACE] 학습 잠금 (Concert Mode)
        
        self.load_plan() 

    def update(self, beat, input_tempo_ratio, input_velocity, confidence):
        """
        [Rehearsal Mode] Update plan based on live input.
        If locked (Concert Mode), updates are ignored.
        """
        # [VIVACE] 잠금 상태면 학습 건너뛰기
        if self.is_locked:
            return

        idx = int(beat)
        if idx < 0 or idx >= len(self.tempo_plan): return

        # Confidence-Gated Update
        alpha = 0.2 * confidence

        # WEMA Update
        self.tempo_plan[idx] = (1 - alpha) * self.tempo_plan[idx] + alpha * input_tempo_ratio
        
        if input_velocity > 0:
            self.dynamics_plan[idx] = (1 - alpha) * self.dynamics_plan[idx] + alpha * input_velocity
            
        self.visit_counts[idx] += 1

    def get_params(self, beat):
        """
        Get current parameters for the given beat.
        """
        idx = int(beat)
        # 범위 체크 및 안전한 반환
        if idx < 0 or idx >= len(self.tempo_plan): 
            return 1.0, 64
            
        return self.tempo_plan[idx], self.dynamics_plan[idx]

    def get_window_size(self, beat):
        """Get adaptive window size for a beat. Returns None if default."""
        idx = int(beat)
        if idx < 0 or idx >= len(self.window_plan):
            return None
        w = self.window_plan[idx]
        return float(w) if w > 0 else None

    def update_window_from_errors(self, beat_errors, default_window=5.0,
                                   error_threshold=2.0, expanded_window=10.0):
        """Update per-beat window sizes based on alignment errors from a rehearsal.

        :param beat_errors: dict {beat_idx: error_in_seconds}
        :param default_window: default window size in seconds
        :param error_threshold: errors above this trigger window expansion
        :param expanded_window: expanded window size in seconds
        """
        if self.is_locked:
            return
        for beat, error in beat_errors.items():
            idx = int(beat)
            if 0 <= idx < len(self.window_plan):
                if error > error_threshold:
                    self.window_plan[idx] = expanded_window
                else:
                    # Shrink back toward default if error is small
                    self.window_plan[idx] = max(self.window_plan[idx] * 0.5, 0)

    def update_from_warping_path(self, warping_path, frame_rate, tempo_bpm,
                                 alpha=0.3):
        """Post-hoc plan update from a completed DTW warping path.

        For each beat, compute the average ref-advancement-per-perf-frame
        (tempo ratio) from the warping path.  A ratio >1 means the
        performer was faster than the reference at that beat; <1 means
        slower.  Blends with the existing plan via WEMA.
        """
        if self.is_locked:
            return

        ref_frames = warping_path[0]
        perf_frames = warping_path[1]
        ref_beats = ref_frames / frame_rate * (tempo_bpm / 60.0)

        max_beat = int(ref_beats[-1]) + 1
        for beat in range(min(max_beat, len(self.tempo_plan))):
            # Skip beats that were explicitly set by intent injection
            if beat < len(self.locked_beats) and self.locked_beats[beat]:
                continue
            mask = (ref_beats >= beat) & (ref_beats < beat + 1)
            if np.sum(mask) < 2:
                continue
            perf_span = float(perf_frames[mask][-1] - perf_frames[mask][0])
            ref_span = float(ref_frames[mask][-1] - ref_frames[mask][0])
            if ref_span == 0:
                continue
            ratio = np.clip(perf_span / ref_span, 0.1, 5.0)
            self.tempo_plan[beat] = (1 - alpha) * self.tempo_plan[beat] + alpha * ratio
            self.visit_counts[beat] += 1

    def save_plan(self):
        """Save current plan to disk"""
        try:
            with open(self.save_path, 'wb') as f:
                pickle.dump((self.tempo_plan, self.dynamics_plan, self.locked_beats), f)
            print(f"[VIVACE] Plan saved to {self.save_path}")
        except Exception as e:
            print(f"[VIVACE] Failed to save plan: {e}")

    def load_plan(self):
        """Load plan from disk"""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'rb') as f:
                    loaded = pickle.load(f)

                # Handle both old (2-tuple) and new (3-tuple) formats
                if len(loaded) == 3:
                    t_plan, d_plan, l_beats = loaded
                else:
                    t_plan, d_plan = loaded
                    l_beats = None

                # 로드된 플랜의 길이를 현재 악보 길이에 맞춤 (Truncate or Pad)
                load_len = min(len(t_plan), len(self.tempo_plan))

                self.tempo_plan[:load_len] = t_plan[:load_len]
                self.dynamics_plan[:load_len] = d_plan[:load_len]
                if l_beats is not None:
                    lock_len = min(len(l_beats), len(self.locked_beats))
                    self.locked_beats[:lock_len] = l_beats[:lock_len]
                
                # 파일에서 로드했다면, 이는 '이미 학습된 플랜(VirtuosoNet)'이므로 잠금
                self.is_locked = True
                print(f"[VIVACE] Loaded & Locked plan from {self.save_path} (Concert Mode)")
                
            except Exception as e:
                print(f"[VIVACE] Failed to load memory: {e}. Starting fresh.")
                self.is_locked = False
        else:
            print("[VIVACE] No existing plan found. Starting in Rehearsal Mode.")
            self.is_locked = False