import whisper
import librosa
import numpy as np
import warnings

# Librosa의 자잘한 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. STT 변환 함수 (Whisper)
# ---------------------------------------------------------
def extract_text_with_whisper(audio_path):
    print("🎙️ Whisper STT 변환 중...")
    # 'base' 모델은 가볍고 빠릅니다. (정확도를 높이려면 'small' 사용)
    model = whisper.load_model("base") 
    result = model.transcribe(audio_path, language="ko")
    return result["text"].strip()

# ---------------------------------------------------------
# 2. 오디오 물리량 추출 함수 (Librosa)
# ---------------------------------------------------------
def extract_audio_features(audio_path):
    print("📊 Librosa 오디오 물리량 분석 중...")
    # sr=None: 원본 샘플링 레이트 유지
    y, sr = librosa.load(audio_path, sr=None)
    
    # ① 음량 (RMS Energy): 목소리의 크기 계산
    rms = librosa.feature.rms(y=y)[0]
    mean_volume = np.mean(rms)
    
    # ② 말하기 속도 (Tempo): 목소리의 빠르기(흥분도) 계산
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # librosa 버전에 따라 tempo가 배열로 반환될 수 있으므로 처리
    mean_tempo = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    
    return mean_volume, mean_tempo

# ---------------------------------------------------------
# 3. 하이브리드 점수 산출 로직
# ---------------------------------------------------------
def calculate_voice_penalty(volume, tempo, vol_threshold=0.05, tempo_threshold=150.0):
    """
    물리량이 특정 임계치(Threshold)를 넘으면 감점을 부여하는 룰 기반 엔진
    """
    penalty = 0
    reasons = []
    
    # 음량이 너무 큰 경우 (고함/Shouting)
    if volume > vol_threshold:
        penalty -= 1
        reasons.append("비정상적인 큰 목소리 감지 (고함)")
        
    # 말이 너무 빠른 경우 (흥분/Agitation)
    if tempo > tempo_threshold:
        penalty -= 1
        reasons.append("비정상적으로 빠른 말하기 감지 (흥분)")
        
    return penalty, reasons

# ---------------------------------------------------------
# 4. 전체 파이프라인 실행 테스트
# ---------------------------------------------------------
if __name__ == "__main__":
    # 테스트할 오디오 파일 경로 (C:\unstructuredData\test.wav)
    target_audio = r"C:\unstructuredData\test.wav"
    
    print("==================================================")
    print(" 🚀 오디오 하이브리드 파이프라인 분석 시작")
    print("==================================================")
    
    try:
        # 1단계: 텍스트 추출
        extracted_text = extract_text_with_whisper(target_audio)
        
        # 2단계: 물리량 추출
        volume, tempo = extract_audio_features(target_audio)
        
        # 3단계: 감점 로직 적용
        penalty, reasons = calculate_voice_penalty(volume, tempo)
        
        # 결과 출력
        print("\n[ 분석 결과 요약 ]")
        print(f"📝 변환된 텍스트: \"{extracted_text}\"")
        print(f"🔊 평균 음량 (Volume): {volume:.4f}")
        print(f"⏱️ 말하기 속도 (Tempo): {tempo:.1f} BPM")
        print("-" * 50)
        
        if penalty < 0:
            print(f"⚠️ [음성 거칠기 감점 발생]: {penalty}점")
            for r in reasons:
                print(f"  - 원인: {r}")
        else:
            print("🟢 [음성 상태 안정적]: 감점 없음")
            
        print("==================================================")
        
    except FileNotFoundError as e:
        if "ffmpeg" in str(e).lower() or "ffprobe" in str(e).lower():
            print("❌ 에러: 시스템에서 FFmpeg를 찾을 수 없습니다. FFmpeg 설치가 완료되었으니, 터미널을 재시작한 후 다시 실행해 주세요.")
        else:
            print("❌ 에러: test.wav 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")