import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
import av
import threading  # NOVO: Necessário para thread-safety

# Configuração da página Web
st.set_page_config(page_title="Scanner de Gabarito PRO", layout="wide")
st.title("🎓 Scanner de Gabaritos - Captura Automática")

# Servidor STUN padrão
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

LETRAS = ["A", "B", "C", "D", "E"]
padrao_letras = [1, 3, 2, 1, 2, 3, 2, 1, 0, 1]  # BDCBCDCBAB

# === ESTADO COMPARTILHADO (Thread-Safe) ===
# NOVO: Lock e dicionário para passar a foto da thread da câmera para a thread da UI
lock = threading.Lock()
container_captura = {"foto": None, "nota": 0}

# === PAINEL DE CONTROLE NA BARRA LATERAL ===
st.sidebar.header("⚙️ Calibração dos Pontos")
v_x1 = st.sidebar.slider("Q01-05: Horiz (X1)", 0, 400, 100)
v_y1 = st.sidebar.slider("Q01-05: Vert (Y1)", 0, 400, 104)
v_x2 = st.sidebar.slider("Q06-10: Horiz (X2)", 0, 800, 356)
v_y2 = st.sidebar.slider("Q06-10: Vert (Y2)", 0, 400, 105)
v_spX = st.sidebar.slider("Largura Letras", 0, 150, 33)
v_spY = st.sidebar.slider("Altura Linhas", 0, 250, 51)

gabarito_list = []
cols = st.sidebar.columns(2)
for i in range(10):
    col = cols[0] if i < 5 else cols[1]
    val = col.selectbox(f"Q{i+1:02d}", LETRAS, index=padrao_letras[i], key=f"q_{i}")
    gabarito_list.append(val)
gabarito_dinamico = "".join(gabarito_list)

# Função auxiliar de ordenação infalível
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

# === PROCESSADOR DE VÍDEO ===
class ScannerGabarito(VideoProcessorBase):
    def __init__(self):
        self.v_x1, self.v_y1 = v_x1, v_y1
        self.v_x2, self.v_y2 = v_x2, v_y2
        self.v_spX, self.v_spY = v_spX, v_spY
        self.gabarito = gabarito_dinamico
        self.snapshot_taken = False  # NOVO: Bandeira para evitar múltiplas capturas

    def update_params(self, x1, y1, x2, y2, spX, spY, gab):
        self.v_x1, self.v_y1 = x1, y1
        self.v_x2, self.v_y2 = x2, y2
        self.v_spX, self.v_spY = spX, spY
        self.gabarito = gab

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # ... (Lógica de pré-processamento IDÊNTICA ao seu código anterior) ...
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
                        anchors.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])

        nota_calculada = 0
        grading_success = False

        if len(anchors) == 4:
            pts_origem = ordenar_pontos(anchors)
            dW, dH = 600, 450
            pts_destino = np.float32([[0, 0], [dW, 0], [0, dH], [dW, dH]])
            M_trans = cv2.getPerspectiveTransform(pts_origem, pts_destino)
            warped = cv2.warpPerspective(img, M_trans, (dW, dH))
            
            # ... (Lógica de Threshold e Amostragem IDÊNTICA) ...
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            _, warped_thresh = cv2.threshold(cv2.GaussianBlur(warped_gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            acertos = 0
            for q in range(10):
                # ... (Calculo de ROIs IDÊNTICO) ...
                is_col2 = q >= 5
                col_base_x = self.v_x2 if is_col2 else self.v_x1
                col_base_y = self.v_y2 if is_col2 else self.v_y1
                linha_idx = (q - 5) if is_col2 else q
                y_ponto = col_base_y + (linha_idx * self.v_spY)
                valores_alternativas = []
                for i in range(5):
                    x_ponto = col_base_x + (i * self.v_spX)
                    if 12 <= y_ponto < dH - 12 and 12 <= x_ponto < dW - 12:
                        roi = warped_thresh[y_ponto-12:y_ponto+12, x_ponto-12:x_ponto+12]
                        valores_alternativas.append(cv2.countNonZero(roi))
                    else:
                        valores_alternativas.append(0)

                idx_marcado = np.argmax(valores_alternativas)
                max_pixels = valores_alternativas[idx_marcado]
                outras_alts = [v for idx, v in enumerate(valores_alternativas) if idx != idx_marcado]
                media_outras = np.mean(outras_alts) if outras_alts else 0

                resposta_aluno = LETRAS[idx_marcado] if (max_pixels > 120 and max_pixels > media_outras * 1.8) else "-"
                if q < len(self.gabarito) and resposta_aluno == self.gabarito[q]:
                    acertos += 1
                grading_success = True

            nota_calculada = (acertos / 10.0) * 10.0
            
            # Feedback visual IDÊNTICO
            for p in anchors: cv2.circle(img, (p[0], p[1]), 8, (0, 255, 0), -1)
            cv2.putText(img, f"Nota: {nota_calculada:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if nota_calculada >= 6 else (0, 0, 255), 2)
            img[0:150, img.shape[1]-200:img.shape[1]] = cv2.resize(warped, (200, 150))

            # === NOVO: LÓGICA DE CAPTURA AUTOMÁTICA ===
            # Gatilho: Enquadramento OK, Grading OK, Foto ainda não tirada para este alinhamento
            if grading_success and self.snapshot_taken == False:
                # Segurança: Usamos o cadeado para escrever no recipiente compartilhado
                with lock:
                    container_captura["foto"] = img.copy()  # Tiramos cópia do frame ATUAL
                    container_captura["nota"] = nota_calculada
                
                self.snapshot_taken = True  # Desativa para o próximo frame
                print(f"-> Captura Automática Realizada! Nota: {nota_calculada}")

        else:
            # Feedback visual IDÊNTICO
            cv2.putText(img, "Alinhe os 4 cantos...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # === NOVO: RESET DA BANDEIRA ===
            # Se enquadramento foi perdido, resetamos a bandeira para permitir nova captura
            self.snapshot_taken = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Inicializa o componente de vídeo
ctx = webrtc_streamer(
    key="snapshot-scanner", 
    rtc_configuration=RTC_CONFIG, 
    video_processor_factory=ScannerGabarito,
    media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False}
)

# Atualiza parâmetros IDÊNTICO
if ctx.video_processor:
    ctx.video_processor.update_params(v_x1, v_y1, v_x2, v_y2, v_spX, v_spY, gabarito_dinamico)

# === MUDANÇA NA UI: EXIBIÇÃO DA FOTO CAPTURADA ===
st.divider()
st.subheader("📸 Última Captura Automática")

# Recipiente vazio para exibir a imagem (slot)
slot_imagem = st.empty()

# NOVO: Mecanismo de Polling (UI verifica constantemente se há foto nova)
foto_para_exibir = None
nota_para_exibir = 0

# Segurança: Cadeado para ler do recipiente compartilhado
with lock:
    if container_captura["foto"] is not None:
        # Pega a foto, mas não tiramos cópia (BGR -> RGB)
        foto_para_exibir = cv2.cvtColor(container_captura["foto"], cv2.COLOR_BGR2RGB)
        nota_para_exibir = container_captura["nota"]

# Exibe na UI se houver algo
if foto_para_exibir is not None:
    slot_imagem.image(foto_para_exibir, caption=f"Foto capturada automaticamente. Nota final calculada: {nota_para_exibir:.1f}", use_column_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"Gabarito corrigido e salvo internamente com Nota {nota_para_exibir:.1f}!")
    with col2:
        # Dica: Você poderia adicionar um st.download_button aqui para salvar a foto
        st.info("Para capturar uma nova foto, mova o gabarito para fora de alinhamento e enquadre-o novamente.")
else:
    slot_imagem.info("Aguardando alinhamento perfeito para captura automática...")
