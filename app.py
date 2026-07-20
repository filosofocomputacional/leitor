import streamlit as st
import cv2
import numpy as np

# Configuração da página Web
st.set_page_config(page_title="Scanner de Gabarito", layout="wide")
st.title("🎓 CorrigePro - Leitor de Gabaritos")
st.write("Tire uma foto do gabarito ou envie uma imagem da sua galeria para realizar a correção.")

LETRAS = ["A", "B", "C", "D", "E"]
padrao_letras = [1, 3, 2, 1, 2, 3, 2, 1, 0, 1]  # Gabarito Padrão (BDCBCDCBAB)

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


# === FUNÇÃO PARA ORDENAR OS 4 CANTOS ===
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
    # Converte os bytes recebidos do navegador para imagem OpenCV
    bytes_data = imagem_para_processar.getvalue()
    file_bytes = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Cópia para desenhar os resultados
    imagem_resultado = img.copy()

    # Pré-processamento
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 11)

    # Busca de contornos dos 4 cantos
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
        # Desenha os pontos nas pontas
        for p in anchors:
            cv2.circle(imagem_resultado, (p[0], p[1]), 10, (0, 255, 0), -1)

        # Ordenação e Transformação de Perspectiva
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

        # Processamento das 10 questões
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

            # Análise Relativa
            idx_marcado = np.argmax(valores_alternativas)
            max_pixels = valores_alternativas[idx_marcado]
            outras_alts = [v for idx, v in enumerate(valores_alternativas) if idx != idx_marcado]
            media_outras = np.mean(outras_alts) if outras_alts else 0

            if max_pixels > 120 and max_pixels > (media_outras * 1.8):
                resposta_aluno = LETRAS[idx_marcado]
            else:
                resposta_aluno = "-"
            
            respostas_aluno.append(resposta_aluno)

            # Marcação na imagem desentortada
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
            st.write(f"**Respostas Lido:** `{''.join(respostas_aluno)}`")

        with col_res2:
            st.subheader("🔍 Ponto de Leitura (Warped)")
            # Mostra a folha perfeitamente retificada com a leitura das bolinhas
            st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption="Análise do Alinhamento e Leitura", use_column_width=True)

    else:
        st.warning("⚠️ **Os 4 cantos do gabarito não foram identificados.**")
        st.info("Dicas: Certifique-se de que a foto está bem iluminada, sem sombras fortes e com os 4 quadrados pretos das pontas bem visíveis.")
        
    st.divider()
    st.subheader("🖼️ Foto Original Processada")
    st.image(cv2.cvtColor(imagem_resultado, cv2.COLOR_BGR2RGB), use_column_width=True)

else:
    st.info("👆 Tire uma foto acima para começar.")
