import subprocess
import pickle
import os
import sys
import glob
import numpy as np
import partitura 

# [설정] VirtuosoNet 환경 및 경로
VIRTUOSO_ENV_PYTHON = "/home/ykim3098/miniconda3/envs/virtuoso/bin/python" 
VIRTUOSO_ROOT = "/home/ykim3098/virtuosoNet"
MODEL_SCRIPT = os.path.join(VIRTUOSO_ROOT, "model_run.py")
MEMORY_FILE = "vivace_memory.pkl"

# 임시 XML 경로
TEMP_INPUT_DIR = os.path.abspath("temp_virtuoso_input")
TEMP_XML_PATH = os.path.join(TEMP_INPUT_DIR, "musicxml_cleaned.musicxml")

def run_virtuoso(mode, args):
    """Subprocess wrapper for VirtuosoNet"""
    cmd = [VIRTUOSO_ENV_PYTHON, MODEL_SCRIPT, f"-mode={mode}"] + args
    
    env = os.environ.copy()
    env["PYTHONPATH"] = VIRTUOSO_ROOT 
    
    # GPU 숨기기 (CPU 모드)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    if "LD_LIBRARY_PATH" in env:
        del env["LD_LIBRARY_PATH"]
        
    print(f"🚀 [VirtuosoBridge] Running in {VIRTUOSO_ROOT} (CPU Mode): {' '.join(cmd)}")
    
    result = subprocess.run(cmd, env=env, cwd=VIRTUOSO_ROOT, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error in virtuosoNet:\n{result.stderr}")
        return False
    
    print("✅ VirtuosoNet execution successful.")
    return True

def parse_virtuoso_output(output_dir):
    """
    VirtuosoNet이 생성한 MIDI를 읽어서 Performance Plan으로 변환
    """
    print(f"📂 Parsing output from: {output_dir}")
    
    midi_files = glob.glob(os.path.join(output_dir, "*.mid"))
    if not midi_files:
        print("❌ No MIDI output found!")
        return

    midi_path = max(midi_files, key=os.path.getmtime)
    print(f"🎹 Found generated MIDI: {os.path.basename(midi_path)}")

    try:
        spart = partitura.load_score(TEMP_XML_PATH)
        ppart = partitura.load_performance(midi_path)
        
        s_notes = spart.note_array()
        p_notes = ppart.note_array()
        
        # 악보 길이 계산 (음수 박자 고려하여 최대값 찾기)
        max_beat = int(np.ceil(s_notes['onset_beat'].max())) + 10
        
        # Plan 배열 초기화
        tempo_plan = np.ones(max_beat)
        dynamics_plan = np.full(max_beat, 64.0)
        
        min_len = min(len(s_notes), len(p_notes))
        
        # 비트별 수집용 딕셔너리
        beat_velocities = {i: [] for i in range(max_beat)}
        
        for i in range(min_len):
            s_note = s_notes[i]
            p_note = p_notes[i]
            
            beat_idx = int(s_note['onset_beat'])
            
            # [수정 핵심] 인덱스가 유효한 범위(0 이상)일 때만 데이터 수집
            if 0 <= beat_idx < max_beat:
                beat_velocities[beat_idx].append(p_note['velocity'])

        # Plan 채우기
        for b in range(max_beat):
            if beat_velocities[b]:
                dynamics_plan[b] = np.mean(beat_velocities[b])
            else:
                if b > 0: dynamics_plan[b] = dynamics_plan[b-1]

        with open(MEMORY_FILE, "wb") as f:
            pickle.dump((tempo_plan, dynamics_plan), f)
        print(f"💾 extracted Plan saved to {MEMORY_FILE} (Max Beat: {max_beat})")
        print(f"   - Average Velocity: {np.mean(dynamics_plan):.1f}")
        
    except Exception as e:
        print(f"❌ Failed to parse/align: {e}")
        # 디버깅을 위해 상세 에러 출력
        import traceback
        traceback.print_exc()

def generate_base_plan(composer="Bach", score_path="./score.musicxml"):
    """Phase 1: Base Generation"""
    print(f"\n🎵 Generating Base Plan ({composer} style)...")
    
    if not os.path.exists(score_path):
        print(f"❌ Score file not found: {score_path}")
        return

    os.makedirs(TEMP_INPUT_DIR, exist_ok=True)
    subprocess.run(["cp", score_path, TEMP_XML_PATH])
    
    path_arg = os.path.join(TEMP_INPUT_DIR, "") 
    
    success = run_virtuoso("test", [
        f"-comp={composer}",
        f"-path={path_arg}", 
        "-bp=true"
    ])
    
    if success:
        parse_virtuoso_output(os.path.join(VIRTUOSO_ROOT, "test_result"))

def refine_style(rehearsal_midi="rehearsal_input.mid"):
    """Phase 2: Style Refinement"""
    print(f"\n✨ Refining Style from {rehearsal_midi}...")
    
    if not os.path.exists(rehearsal_midi):
        print("❌ Rehearsal MIDI file not found!")
        return

    target_path = os.path.join(VIRTUOSO_ROOT, "emotionNet", "rehearsal.mid")
    subprocess.run(["cp", rehearsal_midi, target_path])
    
    success = run_virtuoso("testAll", [
        "-code=isgn", 
        "-bp=true"
    ])
    
    if success:
        parse_virtuoso_output(os.path.join(VIRTUOSO_ROOT, "test_result"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python matchmaker/generate_plan.py [base|refine] [Composer/MidiPath] [ScorePath]")
        sys.exit(1)
        
    action = sys.argv[1]
    
    if action == "base":
        comp = sys.argv[2] if len(sys.argv) > 2 else "Chopin"
        score_input = sys.argv[3] if len(sys.argv) > 3 else "./score.musicxml"
        generate_base_plan(composer=comp, score_path=score_input)
        
    elif action == "refine":
        midi = sys.argv[2] if len(sys.argv) > 2 else "rehearsal_input.mid"
        refine_style(rehearsal_midi=midi)
    
    else:
        print(f"Unknown action: {action}")
        
"""
python matchmaker/generate_plan.py base Chopin /home/ykim3098/virtuosoNet/test_pieces/chopin_nocturne/musicxml_cleaned.musicxml
"""