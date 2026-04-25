import pickle
import mido
import os
import partitura
import numpy as np
from mido import MidiFile, MidiTrack, MetaMessage

# 파일 설정
memory_path = "vivace_memory.pkl"
# 우선순위: 1. MIDI 악보 -> 2. MusicXML 악보 -> 3. 없으면 에러
score_midi_path = "simple_score.mid" 
score_musicxml_path = "matchmaker/assets/simple_score.musicxml" # run_examples.py가 받는 기본 악보 이름
output_midi = "vivace_expressive_output.mid"

def ensure_midi_score():
    """MusicXML만 있다면 MIDI로 변환하여 저장"""
    if os.path.exists(score_midi_path):
        print(f"[VIVACE] Found MIDI score: {score_midi_path}")
        return score_midi_path
    
    if os.path.exists(score_musicxml_path):
        print(f"[VIVACE] Converting MusicXML ({score_musicxml_path}) to MIDI...")
        score = partitura.load_score(score_musicxml_path)
        partitura.save_score_midi(score, score_midi_path)
        return score_midi_path
    
    # 파일 이름을 못 찾을 경우를 대비해 현재 폴더의 .musicxml 검색
    files = [f for f in os.listdir('.') if f.endswith('.musicxml')]
    if files:
        print(f"[VIVACE] Converting detected MusicXML ({files[0]}) to MIDI...")
        score = partitura.load_score(files[0])
        partitura.save_score_midi(score, score_midi_path)
        return score_midi_path

    return None

def apply_vivace_expression():
    # 1. MIDI 준비
    input_midi = ensure_midi_score()
    if not input_midi:
        print("Error: No score file (.mid or .musicxml) found in directory.")
        print("Please verify you ran 'run_examples.py' successfully.")
        return

    # 2. 기억(Brain) 불러오기
    try:
        with open(memory_path, 'rb') as f:
            tempo_plan, dynamics_plan = pickle.load(f)
        print(f"[VIVACE] Brain loaded: {len(tempo_plan)} slots")
    except FileNotFoundError:
        print(f"Error: {memory_path} not found. Run the tracking example first.")
        return

    # 3. 악보에 생명 불어넣기 (Expression Injection)
    mid = MidiFile(input_midi)
    new_mid = MidiFile(ticks_per_beat=mid.ticks_per_beat)
    
    print(f"[VIVACE] Injecting expression into {input_midi}...")

    for i, track in enumerate(mid.tracks):
        new_track = MidiTrack()
        new_mid.tracks.append(new_track)
        
        current_tick = 0
        
        for msg in track:
            # 시간(Delta time)을 절대 시간(Tick)으로 추적하여 현재 비트 계산
            current_beat = int(current_tick / mid.ticks_per_beat)
            
            # --- [Expression 1] Tempo (Rubato) ---
            # 학습된 템포 비율 가져오기 (없으면 1.0)
            tempo_ratio = 1.0
            if current_beat < len(tempo_plan):
                tempo_ratio = tempo_plan[current_beat]
                if tempo_ratio <= 0.1: tempo_ratio = 1.0 # 안전장치
            
            # 델타 타임 변형: 빠르면(ratio>1) 시간을 줄이고, 느리면(ratio<1) 늘림
            new_delta = int(msg.time / tempo_ratio)
            msg.time = new_delta
            current_tick += new_delta

            # --- [Expression 2] Dynamics (Volume) ---
            if msg.type == 'note_on' and msg.velocity > 0:
                learned_vol = 64.0
                if current_beat < len(dynamics_plan):
                    learned_vol = dynamics_plan[current_beat]
                
                # 학습된 값이 유의미할 때만 적용 (기본값 64 제외)
                if abs(learned_vol - 64) > 1:
                    # 기존 50% + 학습 50% 블렌딩
                    new_vel = int(msg.velocity * 0.5 + learned_vol * 0.5)
                    msg.velocity = np.clip(new_vel, 1, 127)

            new_track.append(msg)

    # 4. 결과 저장
    new_mid.save(output_midi)
    print(f"✅ Success! Generated expressive performance: {output_midi}")

if __name__ == "__main__":
    apply_vivace_expression()