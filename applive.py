import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
import av

# Configuração da página Web
st.set_page_config(page_title="Scanner de Gabarito", layout="wide")
st.title("🎓 Scanner de Gabaritos em Tempo Real")

# Servidor STUN (necessário para a câmera funcionar na nuvem/celular via 4G ou Wi-Fi)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

LETRAS = ["A", "B", "C", "D", "E"]
padrao_letras = [1, 3, 2, 1, 2, 3, 2, 1, 0, 1]  # BDCBCDCBAB

# === PAINEL DE CONTROLE NA BARRA LATERAL (SUBSTITUI OS TRACKBARS DO OPENCV) ===
st.sidebar.header("⚙️ Calibração dos Pontos")
v_x1 = st.sidebar.slider("Q01-05: Horizontal (X1)", 0, 400, 100)
v_y1 = st.sidebar.slider("Q01-05: Vertical (Y1)", 0, 400, 104)
v_x2 = st.sidebar.slider("Q06-10: Horizontal (X2)", 0, 800, 356)
v_y2 = st.sidebar.slider("Q06-10: Vertical (Y2)", 0, 400, 105)
v_spX = st.sidebar.slider("Espaço Letras (Largura)", 0, 150, 33)
v_spY = st.sidebar.slider("Espaço Linhas (Altura)", 0, 250, 51)

st.sidebar.header("📝 Gabarito Oficial do Professor")
gabarito_list = []
cols = st.sidebar.columns(2)
for i in range(10):
    col = cols[0] if i < 5 else cols[1]
    val = col.selectbox(f"Q{i+1:02d}", LETRAS, index=padrao_letras[i], key=f"q_{i}")
    gabarito_list.append(val)

gabarito_dinamico = "".join(gabarito_list)

# === PROCESSADOR DE VÍDEO EM TEMPO REAL ===
class ScannerGabarito(VideoProcessorBase):
    def __init__(self):
        self.v_x1 = v_x1
        self.v_y1 = v_y1
        self.v_x2 = v_x2
        self.v_y2 = v_y2
        self.v_spX = v_spX
        self.v_spY = v_spY
        self.gabarito = gabarito_dinamico

    def update_params(self, x1, y1, x2, y2, spX, spY, gab):
        self.v_x1, self.v_y1 = x1, y1
        self.v_x2, self.v_y2 = x2, y2
        self.v_spX, self.v_spY = spX, spY
        self.gabarito = gab

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 11)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        anchors = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if 300 < (w * h) < 30000 and 0.7 < (w / float(h)) < 1.3:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        anchors.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

        if len(anchors) == 4:
            for p in anchors:
                cv2.circle(img, p, 8, (0, 255, 0), -1)
            cv2.putText(img, "Alinhado! Lendo...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Ordena os pontos e faz a transformação de perspectiva
            anchors = sorted(anchors, key=lambda p: p[1])
            top = sorted(anchors[:2], key=lambda p: p[0])
            bot = sorted(anchors[2:], key=lambda p: p[0])
            pts_origem = np.float32([top[0], top[1], bot[0], bot[1]])
            
            dW, dH = 600, 450
            pts_destino = np.float32([[0, 0], [dW, 0], [0, dH], [dW, dH]])
            M_trans = cv2.getPerspectiveTransform(pts_origem, pts_destino)
            
            warped = cv2.warpPerspective(img, M_trans, (dW, dH))
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            _, warped_thresh = cv2.threshold(cv2.GaussianBlur(warped_gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            acertos = 0
            for q in range(10):
                is_col2 = q >= 5
                col_base_x = self.v_x2 if is_col2 else self.v_x1
                col_base_y = self.v_y2 if is_col2 else self.v_y1
                linha_idx = (q - 5) if is_col2 else q
                y_ponto = col_base_y + (linha_idx * self.v_spY)

                valores_alternativas = []
                for i in range(5):
                    x_ponto = col_base_x + (i * self.v_spX)
                    if 0 <= y_ponto < dH and 0 <= x_ponto < dW:
                        roi = warped_thresh[y_ponto-12:y_ponto+12, x_ponto-12:x_ponto+12]
                        total_pixels = cv2.countNonZero(roi)
                    else:
                        total_pixels = 0
                    valores_alternativas.append(total_pixels)

                idx_marcado = np.argmax(valores_alternativas)
                resposta_aluno = LETRAS[idx_marcado] if valores_alternativas[idx_marcado] > 110 else "-"

                if q < len(self.gabarito) and resposta_aluno == self.gabarito[q]:
                    acertos += 1

            nota = (acertos / 10.0) * 10.0
            cv2.putText(img, f"Nota: {nota:.1f} | Acertos: {acertos}/10", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if nota >= 6 else (0, 0, 255), 2)
        else:
            cv2.putText(img, "Alinhe os 4 cantos...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(img, f"Gabarito: {self.gabarito}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Inicializa o componente de vídeo WebRTC
ctx = webrtc_streamer(
    key="gabarito-scanner", 
    rtc_configuration=RTC_CONFIG, 
    video_processor_factory=ScannerGabarito
)

# Atualiza os parâmetros em tempo real conforme você mexe na barra lateral
if ctx.video_processor:
    ctx.video_processor.update_params(v_x1, v_y1, v_x2, v_y2, v_spX, v_spY, gabarito_dinamico)
