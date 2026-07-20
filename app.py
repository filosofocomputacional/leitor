import streamlit as st
import cv2
import numpy as np

# Configuração da página Web
st.set_page_config(page_title="Scanner de Gabarito PRO", layout="wide")
st.title("🎓 CorrigePro - Leitor de Gabaritos")

LETRAS = ["A", "B", "C", "D", "E"]
padrao_letras = [1, 3, 2, 1, 2, 3, 2, 1, 0, 1]  # BDCBCDCBAB

# === INICIALIZAÇÃO DO ESTADO DOS SLIDERS (SESSION STATE) ===
if "v_x1" not in st.session_state: st.session_state.v_x1 = 100
if "v_y1" not in st.session_state: st.session_state.v_y1 = 104
if "v_x2" not in st.session_state: st.session_state.v_x2 = 356
if "v_y2" not in st.session_state: st.session_state.v_y2 = 105
if "v_spX" not in st.session_state: st.session_state.v_spX = 33
if "v_spY" not in st.session_state: st.session_state.v_spY = 51

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

# Sliders integrados ao st.session_state
v_x1 = st.sidebar.slider("Q01-05: Horiz (X1)", 0, 400, key="v_x1")
v_y1 = st.sidebar.slider("Q01-05: Vert (Y1)", 0, 400, key="v_y1")
v_x2 = st.sidebar.slider("Q06-10: Horiz (X2)", 0, 800, key="v_x2")
v_y2 = st.sidebar.slider("Q06-10: Vert (Y2)", 0, 400, key="v_y2")
v_spX = st.sidebar.slider("Largura Letras", 0, 150, key="v_spX")
v_spY = st.sidebar.slider("Altura Linhas", 0, 250, key="v_spY")


# === FUNÇÕES AUXILIARES ===
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

def autodetectar_parametros(warped_gray):
    """Detecta círculos usando a Transformada de Hough e calcula os parâmetros dos sliders"""
    blur = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    
    # Busca por formas circulares na imagem desentortada
    circles = cv2.HoughCircles(
        blur, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=18, 
        param1=50, 
        param2=22, 
        minRadius=7, 
        maxRadius=18
    )
    
    if circles is None:
        return None
        
    pts = circles[0, :, :2]
    if len(pts) < 10:
        return None
        
    # Separa os círculos entre Coluna 1 (X < 300) e Coluna 2 (X >= 300)
    col1 = pts[pts[:, 0] < 300]
    col2 = pts[pts[:, 0] >= 300]
    
    if len(col1) < 4 or len(col2) < 4:
        return None

    # Ordena por altura (Y)
    col1_by_y = col1[np.argsort(col1[:, 1])]
    col2_by_y = col2[np.argsort(col2[:, 1])]
    
    # Pega as primeiras bolinhas do topo
    top_col1 = col1_by_y[:5]
    top_col1_a = top_col1[np.argmin(top_col1[:, 0])] # Menor X da linha 1
    
    top_col2 = col2_by_y[:5]
    top_col2_a = top_col2[np.argmin(top_col2[:, 0])]
    
    new_x1, new_y1 = int(top_col1_a[0]), int(top_col1_a[1])
    new_x2, new_y2 = int(top_col2_a[0]), int(top_col2_a[1])
    
    # Estima o espaçamento X (largura)
    top_col1_sorted = top_col1[np.argsort(top_col1[:, 0])]
    if len(top_col1_sorted) > 1:
        diffs_x = np.diff(top_col1_sorted[:, 0])
        valid_diffs = [d for d in diffs_x if 15 <= d <= 60]
        new_spX = int(np.mean(valid_diffs)) if valid_diffs else 33
    else:
        new_spX = 33
        
    # Estima o espaçamento Y (altura entre linhas)
    bot_col1 = col1_by_y[-5:]
    y_top = np.mean(top_col1[:, 1])
    y_bot = np.mean(bot_col1[:, 1])
    new_spY = int((y_bot - y_top) / 4.0) if (y_bot > y_top) else 51
    
    # Limita valores dentro do alcance dos sliders
    new_spY = max(20, min(150, new_spY))
    new_spX = max(15, min(100, new_spX))
    
    return new_x1, new_y1, new_x2, new_y2, new_spX, new_spY


# === OPÇÕES DE CAPTURA ===
aba_camera, aba_upload = st.tabs(["📸 Tirar Foto com Câmera", "📁 Enviar Imagem da Galeria"])
imagem_para_processar = None

with aba_camera:
    foto_camera = st.camera_input("Tire a foto focando bem no gabarito")
    if foto_camera is not None:
        imagem_para_processar = foto_camera

with aba_upload:
    foto_upload = st.file_uploader("Escolha uma foto do seu dispositivo", type=["jpg", "jpeg", "png"])
    if foto_upload is not None:
        imagem_para_processar = foto_upload


# === PROCESSAMENTO DA IMAGEM ===
if imagem_para_processar is not None:
    bytes_data = imagem_para_processar.getvalue()
    file_bytes = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    imagem_resultado = img.copy()

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
            if 300 < (w * h) < 50000 and 0.7 < (w / float(h)) < 1.3:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    anchors.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])

    if len(anchors) == 4:
        for p in anchors:
            cv2.circle(imagem_resultado, (p[0], p[1]), 10, (0, 255, 0), -1)

        pts_origem = ordenar_pontos(anchors)
        dW, dH = 600, 450
        pts_destino = np.float32([[0, 0], [dW, 0], [0, dH], [dW, dH]])
        M_trans = cv2.getPerspectiveTransform(pts_origem, pts_destino)
        
        warped = cv2.warpPerspective(img, M_trans, (dW, dH))
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # === BOTÃO DE AUTODETECÇÃO DE BOLINHAS ===
        col_btn1, col_btn2 = st.columns([1, 2])
        with col_btn1:
            if st.button("🤖 Autodetectar Calibração", type="primary"):
                res = autodetectar_parametros(warped_gray)
                if res is not None:
                    st.session_state.v_x1, st.session_state.v_y1, st.session_state.v_x2, st.session_state.v_y2, st.session_state.v_spX, st.session_state.v_spY = res
                    st.success("Bolinhas identificadas! Sliders ajustados automaticamente.")
                    st.rerun()
                else:
                    st.warning("Não foi possível autodetectar todas as bolinhas. Tente ajustar manualmente.")

        _, warped_thresh = cv2.threshold(
            cv2.GaussianBlur(warped_gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        acertos = 0
        respostas_aluno = []

        # Processamento das 10 questões usando os valores dos sliders
        for q in range(10):
            is_col2 = q >= 5
            col_base_x = v_x2 if is_col2 else v_x1
            col_base_y = v_y2 if is_col2 else v_y1
            linha_idx = (q - 5) if is_col2 else q
            y_ponto = col_base_y + (linha_idx * v_spY)

            valores_alternativas = []
            coords = []

            for i in range(5):
                x_ponto = col_base_x + (i * v_spX)
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

            cor_acerto = (0, 255, 0) if resposta_aluno == gabarito_dinamico[q] else (0, 0, 255)
            for i, (cx, cy) in enumerate(coords):
                r_color = cor_acerto if i == idx_marcado and resposta_aluno != "-" else (200, 200, 200)
                cv2.circle(warped, (cx, cy), 4, r_color, -1)

            if resposta_aluno == gabarito_dinamico[q]:
                acertos += 1

        nota = (acertos / 10.0) * 10.0

        # === EXIBIÇÃO DOS RESULTADOS ===
        st.divider()
        col_res1, col_res2 = st.columns([1, 1])

        with col_res1:
            st.subheader("📊 Resultado da Correção")
            if nota >= 6.0:
                st.success(f"### NOTA: {nota:.1f} / 10.0  ({acertos} acertos)")
            else:
                st.error(f"### NOTA: {nota:.1f} / 10.0  ({acertos} acertos)")

            st.write(f"**Gabarito Oficial:** `{gabarito_dinamico}`")
            st.write(f"**Respostas Lidas:** `{''.join(respostas_aluno)}`")

        with col_res2:
            st.subheader("🔍 Ponto de Leitura (Warped)")
            st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption="Alinhamento Atual dos Sliders", use_column_width=True)

    else:
        st.warning("⚠️ **Os 4 cantos do gabarito não foram identificados.**")
        
    st.divider()
    st.subheader("🖼️ Foto Original Processada")
    st.image(cv2.cvtColor(imagem_resultado, cv2.COLOR_BGR2RGB), use_column_width=True)

else:
    st.info("👆 Tire uma foto acima para começar.")