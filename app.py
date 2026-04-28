import streamlit as st
import pandas as pd
import time
import os

# 방금 만든 오디오 파이프라인 모듈 임포트! (파일명이 audioPipeline.py 인 경우)
try:
    from audioPipeline import extract_text_with_whisper, extract_audio_features, calculate_voice_penalty
except ImportError:
    st.error("오류: 같은 폴더에 audioPipeline.py 파일이 있는지 확인해 주세요!")

# --- [1] 기본 웹 페이지 설정 ---
st.set_page_config(page_title="CS 위기 조기 경보", page_icon="🚨", layout="wide")
st.title("🚨 음성-텍스트 하이브리드 CS 위기 모니터링 시스템")

# --- [2] 세션 상태 초기화 (대화 기록 저장용) ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- [3] 화면 레이아웃 (좌측: 입력 및 대화록 / 우측: 실시간 모니터링) ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎙️ 고객 음성 입력")
    # 파일 업로더 생성
    uploaded_file = st.file_uploader("고객의 음성 파일(.wav)을 업로드하세요", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        # 업로드된 파일을 임시 저장
        temp_audio_path = os.path.join("temp_audio.wav")
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 분석 실행 버튼
        if st.button("분석 시작", type="primary", use_container_width=True):
            with st.spinner("AI가 음성을 분석하고 있습니다..."):
                # 1. 텍스트 추출 (Whisper)
                text = extract_text_with_whisper(temp_audio_path)
                
                # 2. 물리량 추출 (Librosa)
                vol, tempo = extract_audio_features(temp_audio_path)
                
                # 3. 감점 계산
                # (테스트를 위해 임계치를 조금 넉넉하게 0.08로 잡았습니다)
                penalty, reasons = calculate_voice_penalty(vol, tempo, vol_threshold=0.08, tempo_threshold=150.0)
                
                # --- (Mock) 텍스트 감정 분석 ---
                # 주의: 빠른 테스트를 위해 텍스트 점수는 일단 임의로 계산합니다. 
                # 나중에 우리가 만든 KcELECTRA 모델을 여기에 붙일 겁니다!
                text_score = 0
                if "짜증" in text or "화가" in text or "환불" in text:
                    text_score = -2
                elif "걱정" in text or "안 와서" in text:
                    text_score = -1
                    
                total_score = text_score + penalty
                
                # 기록 저장
                st.session_state.history.append({
                    "Turn": len(st.session_state.history) + 1,
                    "Text": text,
                    "Volume": round(vol, 4),
                    "Tempo": round(tempo, 1),
                    "Penalty": penalty,
                    "Total Score": total_score
                })
            st.success("분석 완료!")

    st.divider()
    st.subheader("💬 실시간 대화록")
    for chat in st.session_state.history:
        with st.chat_message("user"):
            st.write(f"**고객:** {chat['Text']}")
            st.caption(f"🔊 음량: {chat['Volume']} | ⏱️ 속도: {chat['Tempo']} BPM | 📉 거칠기 감점: {chat['Penalty']}점")

with col2:
    st.subheader("📈 실시간 대화 위험도 추적")
    
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        df["누적 점수"] = df["Total Score"].cumsum()
        
        # 꺾은선 그래프 렌더링
        st.line_chart(df.set_index("Turn")["누적 점수"], height=300)
        
        current_score = df["누적 점수"].iloc[-1]
        
        st.subheader("🚦 현재 위기 상태")
        # 컬럼을 나누어 현재 지표들 예쁘게 표시
        m1, m2, m3 = st.columns(3)
        m1.metric("현재 턴 누적 점수", current_score)
        m2.metric("최근 음량", df["Volume"].iloc[-1])
        m3.metric("최근 속도", df["Tempo"].iloc[-1])
        
        # 임계점 경고 로직
        if current_score <= -3:
            st.error("### 🚨 [위험] 폭발 임계점 돌파!\n고객의 감정과 목소리가 매우 격앙되었습니다. **즉시 팀장급 상담원 개입을 권고합니다.**")
        elif current_score < 0:
            st.warning("### 🟡 [주의] 대화 악화 중\n고객의 불만이 감지되었습니다. 무조건적인 공감과 사과 멘트를 먼저 사용하세요.")
        else:
            st.success("### 🟢 [안전] 안정적인 대화")
            
        if st.button("초기화", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("좌측에서 음성 파일을 업로드하고 분석을 시작해 보세요.")