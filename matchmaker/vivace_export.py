import os
import pickle
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import partitura

# --- 경로 설정 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")

sf2_path = os.path.join(assets_dir, "default.sf2")
xml_path = os.path.join(assets_dir, "simple_score.musicxml")
midi_path = os.path.join(assets_dir, "simple_score.mid")

memory_path = "vivace_memory.pkl"
expressive_midi = "vivace_expressive_output.mid"
wav_original = "original.wav"
wav_expressive = "vivace_expressive.wav"

def check_files():
    # 1. 사운드폰트
    if not os.path.exists(sf2_path):
        print(f"❌ Error: SoundFont missing at {sf2_path}")
        return False
    
    # 2. 악보 (XML -> MIDI 변환)
    if not os.path.exists(midi_path):
        if os.path.exists(xml_path):
            print(f"🔄 Converting MusicXML to MIDI...")
            try:
                score = partitura.load_score(xml_path)
                partitura.save_score_midi(score, midi_path)
            except Exception as e:
                print(f"❌ XML Conversion failed: {e}")
                return False
        else:
            print(f"❌ Score missing: {xml_path}")
            return False
    return True

def render_audio(midi_in, wav_out):
    if not os.path.exists(midi_in):
        return

    print(f"🎵 Rendering {wav_out}...")
    
    # [핵심 수정] 옵션(-F, -r)을 파일 경로보다 앞에 배치
    cmd = [
        "fluidsynth", 
        "-ni", 
        "-F", wav_out, 
        "-r", "44100", 
        "-g", "1.0", 
        sf2_path, 
        midi_in
    ]
    
    # 실행 (로그 숨김 X -> 에러 확인을 위해 로그 출력 허용)
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if os.path.exists(wav_out) and os.path.getsize(wav_out) > 1000:
        print(f"   ✅ Saved: {wav_out}")
    else:
        print(f"   ❌ Failed. FluidSynth Error Log:\n{result.stderr[:500]}...")

def visualize():
    if not os.path.exists(memory_path):
        return
    print("📊 Generating Graph...")
    try:
        with open(memory_path, 'rb') as f:
            tempo_plan, dynamics_plan = pickle.load(f)
            
        learned_len = len(np.where(tempo_plan != 1.0)[0])
        display_len = min(len(tempo_plan), learned_len + 5) if learned_len > 0 else 20
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        ax1.plot(tempo_plan[:display_len], 'b-o', label='VIVACE Tempo')
        ax1.axhline(1.0, color='r', linestyle='--')
        ax1.set_title(f'VIVACE Analysis (Learned {learned_len} beats)')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2.plot(dynamics_plan[:display_len], 'g-s', label='VIVACE Dynamics')
        ax2.axhline(64, color='r', linestyle='--')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("vivace_analysis.png")
        print("   ✅ Graph saved: vivace_analysis.png")
    except Exception as e:
        print(f"   ⚠️ Graph error: {e}")

def main():
    print("--- [ VIVACE Export Manager v2 ] ---")
    if not check_files(): return
    visualize()
    render_audio(midi_path, wav_original)
    if os.path.exists(expressive_midi):
        render_audio(expressive_midi, wav_expressive)
    else:
        print(f"ℹ️ Expressive MIDI missing. (Run 'vivace_generate.py')")

if __name__ == "__main__":
    main()
