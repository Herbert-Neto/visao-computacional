import cv2
import numpy as np

TINY = False

ARQUIVO_CFG = "trafego/yolov3{}.cfg".format("-tiny" if TINY else "")
ARQUIVO_PESOS = "trafego/yolov3{}.weights".format("-tiny" if TINY else "")
ARQUIVO_CLASSES = "trafego/coco{}.names".format("-tiny" if TINY else "")
video = "trafego/946.mp4"

def carregar_modelo_pretreinado():
    modelo = cv2.dnn.readNetFromDarknet(ARQUIVO_CFG, ARQUIVO_PESOS)
    modelo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    modelo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    if modelo.empty():
        raise IOError("Não foi possível carregar o modelo de detecção de objetos.")
    return modelo

def carregar_classes():
    with open(ARQUIVO_CLASSES, "r") as arquivo:
        classes = [linha.strip() for linha in arquivo.readlines()]
    return classes

def preprocessar_frame(frame):
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    return blob

def detectar_objetos(frame, modelo):
    blob = preprocessar_frame(frame)
    modelo.setInput(blob)
    nomes_camadas = modelo.getLayerNames()
    camadas_saida = [nomes_camadas[i - 1] for i in modelo.getUnconnectedOutLayers()]
    saidas = modelo.forward(camadas_saida)
    return saidas

def processar_frame(modelo, frame, ids_desejados):
    deteccoes = detectar_objetos(frame, modelo)
    ids_classes = []
    confiancas = []
    caixas = []
    altura, largura, _ = frame.shape

    for saida in deteccoes:
        for deteccao in saida:
            deteccao = deteccao.reshape(-1, 85)
            for obj in deteccao:
                pontuacoes = obj[5:]
                id_classe = np.argmax(pontuacoes)
                confianca = pontuacoes[id_classe]
                if confianca > 0.5 and id_classe in ids_desejados:
                    centro_x = int(obj[0] * largura)
                    centro_y = int(obj[1] * altura)
                    w = int(obj[2] * largura)
                    h = int(obj[3] * altura)

                    x = int(centro_x - w / 2)
                    y = int(centro_y - h / 2)

                    caixas.append([x, y, w, h])
                    confiancas.append(float(confianca))
                    ids_classes.append(id_classe)
    return ids_classes, confiancas, caixas

def desenhar_caixas(frame, caixas, confiancas, ids_classes, classes):
    indices = cv2.dnn.NMSBoxes(caixas, confiancas, 0.5, 0.4)

    contador_carros = 0
    contador_motos = 0
    contador_caminhoes = 0
    contador_semaforos = 0
    contador_placas = 0

    if len(indices) > 0:
        indices = indices.flatten()  
        for i in indices:
            caixa = caixas[i]
            x, y, w, h = caixa
            rotulo = str(classes[ids_classes[i]])

            if rotulo == "carro":
                contador_carros += 1
            elif rotulo == "motocicleta":
                contador_motos += 1
            elif rotulo == "caminhao":
                contador_caminhoes += 1
            elif rotulo == "semaforo":
                contador_semaforos += 1
            elif rotulo == "placa de pare":
                contador_placas += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, rotulo, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"Carros: {contador_carros}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Motos: {contador_motos}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Caminhões: {contador_caminhoes}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Semáforos: {contador_semaforos}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Placas de Pare: {contador_placas}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

def main():
    modelo = carregar_modelo_pretreinado()
    classes = carregar_classes()

    ids_desejados = [
        classes.index("carro"),
        classes.index("motocicleta"),
        classes.index("caminhao"),
        classes.index("semaforo"),
        classes.index("placa de pare")
    ]

    caminho_video = video
    cap = cv2.VideoCapture(caminho_video)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ids_classes, confiancas, caixas = processar_frame(modelo, frame, ids_desejados)

        frame = desenhar_caixas(frame, caixas, confiancas, ids_classes, classes)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
