import streamlit as st
import cv2
import numpy as np
import threading
import av
import time  # Adicionado para controlar o temporizador de 2 segundos
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Configuração da página Web
st.set_page_config(page_title="Scanner de Gabarito PRO", layout="wide")
st.title("🎓 CorrigePro - Leitor de Gabaritos")

LETRAS = ["A", "B", "C", "D", "E"]
padrao_letras = [1, 3, 2, 1, 2, 3, 2, 1, 0, 1]  # Gabarito Padrão (BDCBCDCBAB)

# Configuração STUN para conexão mobile
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# === ESTADO DA SESSÃO ===
if "dados_captura" not in st.session_state:
    st.session_state.dados_captura = None

lock = threading.Lock()
if "buffer_captura" not in st.session_state:
    st.session_state.buffer_captura = {"dados": None}

# === BARRA LATERAL: CONTROLES E CALIBRAÇÃO ===
st.sidebar.header("📝 Gabarito Oficial do Professor")
gabarito_list = []
cols = st.sidebar.columns(2)
for i in range(10):
    col = cols[0] if i < 5 else cols[1]
    val = col.selectbox(f"Q{i+1:02d}", LETRAS, index=padrao_letras[i], key=f"q_{i}")
    gabarito_list.append(val)
gabarito_dinamico = "".join(gabarito_list)

st.sidebar.divider()
st.sidebar.header("⚙️ Calibração dos Pontos")
v_x1 = st.sidebar.slider("Q01-05: Horiz (X1)", 0, 400, 100)
v_y1 = st.sidebar.slider("Q01-05: Vert (Y1)", 0, 400, 104)
v_x2 = st.sidebar.slider("Q06-10: Horiz (X2)", 0, 800, 356)
v_y2 = st.sidebar.slider("Q06-10: Vert (Y2)", 0, 400, 105)
v_spX = st.sidebar.slider("Largura Letras", 0, 150, 33)
v_spY = st.sidebar.slider("Altura Linhas", 0, 250, 51)


def ordenar_pontos(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # Top-Left
    rect[3] = pts[np.argmax(s)]      # Bottom-Right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # Top-Right
    rect[2] = pts[np.argmax(diff)]   # Bottom-Left
    return rect


# === PROCESSADOR DE VÍDEO COM TEMPORIZADOR DE 2 SEGUNDOS ===
class ScannerGabarito(VideoProcessorBase):
    def __init__(self):
        self.v_x1, self.v_y1 = v_x1, v_y1
        self.v_x2, self.v_y2 = v_x2, v_y2
        self.v_spX, self.v_spY = v_spX, v_spY
        self.gabarito = gabarito_dinamico
        self.tempo_inicio_alinhado = None  # Marca quando os 4 pontos foram vistos
        self.capturado = False

    def update_params(self, x1, y1, x2, y2, spX, spY, gab):
        self.v_x1, self.v_y1 = x1, y1
        self.v_x2, self.v_y2 = x2, y2
        self.v_spX, self.v_spY = spX, spY
        self.gabarito = gab

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        imagem_exibicao = img.copy()

        # Pré-processamento
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 11)

        # Encontra contornos dos 4 cantos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        anchors = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if 300 < (w * h) < 50000 and 0.7 < (w / float(h)) < 1.3:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        anchors.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])

        if len(anchors) == 4:
            # Se for o primeiro frame alinhado, inicia o cronômetro
            if self.tempo_inicio_alinhado is None:
                self.tempo_inicio_alinhado = time.time()

            tempo_decorrido = time.time() - self.tempo_inicio_alinhado
            tempo_restante = max(0.0, 2.0 - tempo_decorrido)

            # Desenha pontos de alinhamento
            for p in anchors:
                cv2.circle(imagem_exibicao, (p[0], p[1]), 10, (0, 255, 0), -1)

            # Se ainda está no período de estabilização (0 a 2 segundos)
            if tempo_restante > 0:
                cv2.putText(imagem_exibicao, f"SEGURE FIRME! Capturando em {tempo_restante:.1f}s", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(imagem_exibicao, "FOTO CAPTURADA!", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Dispara a captura após 2 segundos estáveis
                if not self.capturado:
                    pts_origem = ordenar_pontos(anchors)
                    dW, dH = 600, 450
                    pts_destino = np.float32([[0, 0], [dW, 0], [0, dH], [dW, dH]])
                    M_trans = cv2.getPerspectiveTransform(pts_origem, pts_destino)
                    
                    warped = cv2.warpPerspective(img, M_trans, (dW, dH))
                    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    _, warped_thresh = cv2.threshold(
                        cv2.GaussianBlur(warped_gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )

                    acertos = 0
                    respostas_aluno = []

                    for q in range(10):
                        is_col2 = q >= 5
                        col_base_x = self.v_x2 if is_col2 else self.v_x1
                        col_base_y = self.v_y2 if is_col2 else self.v_y1
                        linha_idx = (q - 5) if is_col2 else q
                        y_ponto = col_base_y + (linha_idx * self.v_spY)

                        valores_alternativas = []
                        coords = []

                        for i in range(5):
                            x_ponto = col_base_x + (i * self.v_spX)
                            coords.append((x_ponto, y_ponto))
                            
                            if 12 <= y_ponto < dH - 12 and 12 <= x_ponto < dW - 12:
                                roi = warped_thresh[y_ponto-12:y_ponto+12, x_ponto-12:x_ponto+12]
                                total_pixels = cv2.countNonZero(roi)
                            else:
                                total_pixels = 0
                            valores_alternativas.append(total_pixels)

                        idx_marcado = np.argmax(valores_alternativas)
                        max_pixels = valores_alternativas[idx_marcado]
                        outras_alts = [v for idx, v in enumerate(valores_alternativas) if idx != idx_marcado]
                        media_outras = np.mean(outras_alts) if outras_alts else 0

                        if max_pixels > 120 and max_pixels > (media_outras * 1.8):
                            resposta_aluno = LETRAS[idx_marcado]
                        else:
                            resposta_aluno = "-"
                        
                        respostas_aluno.append(resposta_aluno)

                        cor_acerto = (0, 255, 0) if resposta_aluno == self.gabarito[q] else (0, 0, 255)
                        for i, (cx, cy) in enumerate(coords):
                            r_color = cor_acerto if i == idx_marcado and resposta_aluno != "-" else (200, 200, 200)
                            cv2.circle(warped, (cx, cy), 4, r_color, -1)

                        if resposta_aluno == self.gabarito[q]:
                            acertos += 1

                    nota = (acertos / 10.0) * 10.0

                    # Armazena os dados processados no buffer
                    with lock:
                        st.session_state.buffer_captura["dados"] = {
                            "img_original": imagem_exibicao.copy(),
                            "warped": warped.copy(),
                            "nota": nota,
                            "acertos": acertos,
                            "respostas_aluno": respostas_aluno,
                            "gabarito_oficial": self.gabarito
                        }
                    self.capturado = True

        else:
            # Se perder o alinhamento antes dos 2s, reseta o cronômetro
            self.tempo_inicio_alinhado = None
            cv2.putText(imagem_exibicao, "Alinhe os 4 cantos do gabarito...", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(imagem_exibicao, format="bgr24")


# === INTERFACE DO USUÁRIO ===

# Transfere do buffer para a sessão
with lock:
    if st.session_state.buffer_captura["dados"] is not None and st.session_state.dados_captura is None:
        st.session_state.dados_captura = st.session_state.buffer_captura["dados"]

# 1. MODO CÂMERA AO VIVO
if st.session_state.dados_captura is None:
    st.info("📷 **Mantenha o gabarito firme no enquadramento por 2 segundos** para realizar a captura automática.")
    
    ctx = webrtc_streamer(
        key="gabarito-timer-capture", 
        rtc_configuration=RTC_CONFIG, 
        video_processor_factory=ScannerGabarito,
        media_stream_constraints={
            "video": {
                "facingMode": "environment",
                "width": {"ideal": 1280}, 
                "height": {"ideal": 720}
            }, 
            "audio": False
        }
    )

    if ctx.video_processor:
        ctx.video_processor.update_params(v_x1, v_y1, v_x2, v_y2, v_spX, v_spY, gabarito_dinamico)

    with lock:
        if st.session_state.buffer_captura["dados"] is not None:
            st.rerun()

# 2. MODO CONFIRMAÇÃO / RESULTADOS
else:
    dados = st.session_state.dados_captura

    st.subheader("🧐 A captura da área ficou correta?")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 REFAZER FOTO", type="secondary", use_container_width=True):
            with lock:
                st.session_state.buffer_captura["dados"] = None
            st.session_state.dados_captura = None
            st.rerun()

    with col_btn2:
        if st.button("✅ ACEITAR E ESCANEAR PRÓXIMO", type="primary", use_container_width=True):
            with lock:
                st.session_state.buffer_captura["dados"] = None
            st.session_state.dados_captura = None
            st.rerun()

    st.divider()
    col_res1, col_res2 = st.columns([1, 1])

    with col_res1:
        st.subheader("📊 Resultado Lido")
        nota = dados["nota"]
        acertos = dados["acertos"]
        
        if nota >= 6.0:
            st.success(f"### NOTA: {nota:.1f} / 10.0  ({acertos} acertos)")
        else:
            st.error(f"### NOTA: {nota:.1f} / 10.0  ({acertos} acertos)")

        st.write(f"**Gabarito Oficial:** `{dados['gabarito_oficial']}`")
        st.write(f"**Respostas Lidas:** `{''.join(dados['respostas_aluno'])}`")

    with col_res2:
        st.subheader("🔍 Área Recortada e Pontos Lidos")
        # Exibe a folha retificada para o usuário conferir se as bolinhas amarelas/verdes bateram
        st.image(cv2.cvtColor(dados["warped"], cv2.COLOR_BGR2RGB), caption="Confira se os círculos azuis/verdes caíram dentro das bolinhas.", use_column_width=True)

    st.divider()
    st.subheader("🖼️ Foto Completa Capturada")
    st.image(cv2.cvtColor(dados["img_original"], cv2.COLOR_BGR2RGB), use_column_width=True)
