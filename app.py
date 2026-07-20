import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
import av

# Função para ordenar os 4 pontos de forma infalível
def ordenar_pontos(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # Top-Left (menor soma)
    rect[3] = pts[np.argmax(s)]      # Bottom-Right (maior soma)

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # Top-Right (menor diferença)
    rect[2] = pts[np.argmax(diff)]   # Bottom-Left (maior diferença)

    return rect

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
                        anchors.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])

        if len(anchors) == 4:
            for p in anchors:
                cv2.circle(img, (p[0], p[1]), 8, (0, 255, 0), -1)
            cv2.putText(img, "Gabarito Identificado!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 1. Ordenação Infalível
            pts_origem = ordenar_pontos(anchors)
            
            dW, dH = 600, 450
            pts_destino = np.float32([[0, 0], [dW, 0], [0, dH], [dW, dH]])
            M_trans = cv2.getPerspectiveTransform(pts_origem, pts_destino)
            
            warped = cv2.warpPerspective(img, M_trans, (dW, dH))
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            
            # Limpeza e Threshold na imagem retificada
            _, warped_thresh = cv2.threshold(
                cv2.GaussianBlur(warped_gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            acertos = 0
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

                # 2. Análise Relativa da Marcação
                idx_marcado = np.argmax(valores_alternativas)
                max_pixels = valores_alternativas[idx_marcado]
                
                # Média dos pixels das OUTRAS alternativas que não foram marcadas
                outras_alts = [v for idx, v in enumerate(valores_alternativas) if idx != idx_marcado]
                media_outras = np.mean(outras_alts) if outras_alts else 0

                # Só considera marcado se o ponto ativo for claramente maior que a média do resto
                if max_pixels > 120 and max_pixels > (media_outras * 1.8):
                    resposta_aluno = LETRAS[idx_marcado]
                else:
                    resposta_aluno = "-"

                # Desenha feedback visual direto na imagem
                color = (0, 255, 0) if (q < len(self.gabarito) and resposta_aluno == self.gabarito[q]) else (0, 0, 255)
                for i, (cx, cy) in enumerate(coords):
                    r_color = (0, 255, 0) if i == idx_marcado and resposta_aluno != "-" else (255, 255, 255)
                    cv2.circle(warped, (cx, cy), 3, r_color, -1)

                if q < len(self.gabarito) and resposta_aluno == self.gabarito[q]:
                    acertos += 1

            nota = (acertos / 10.0) * 10.0
            cv2.putText(img, f"Nota: {nota:.1f} | Acertos: {acertos}/10", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if nota >= 6 else (0, 0, 255), 2)
            
            # Subtitui um pedaço da imagem para você ver a retificação no canto (Feedback Visual)
            img[0:150, img.shape[1]-200:img.shape[1]] = cv2.resize(warped, (200, 150))

        else:
            cv2.putText(img, "Alinhe os 4 cantos...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. Forçar câmera em Alta Resolução (720p)
ctx = webrtc_streamer(
    key="gabarito-scanner", 
    rtc_configuration=RTC_CONFIG, 
    video_processor_factory=ScannerGabarito,
    media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False}
)
