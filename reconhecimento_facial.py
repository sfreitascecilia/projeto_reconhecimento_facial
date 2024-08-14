import face_recognition
import cv2
# import numpy as np


# Função para carregar uma imagem e obter os embeddings faciais
# (representações vetoriais das imagens)
def load_image_and_get_face_encodings(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encod = face_recognition.face_encodings(image)
    return face_encod


# Carregar a imagem de referência e obter a codificação facial
known_face_image_path = "rosto_conhecido.jpg"  # Imagem de referência
known_face_encodings = load_image_and_get_face_encodings(known_face_image_path)

# Iniciar a captura de vídeo da câmera
video_capture = cv2.VideoCapture(0)  # 0 para abrir a câmera padrão do sistema

while True:
    # Capturar frame a frame
    # ret = indica se a captura foi bem-sucedida
    # frame = imagem capturada
    ret, frame = video_capture.read()
    if not ret:
        print("Não foi possível capturar o vídeo!")
        break

    # Converter o frame da câmera de BGR (opencv-python) para RGB (face_recognition)
    # Essa conversão é necessária devido às diferenças entre como as bibliotecas
    # processam as imagens. A opencv-python usa o formato BGR (Blue, Green, Red) por padrão
    # para capturar imagens e a face_recognition utiliza o formato RGB (Red, Green, Blue) para processar imagens.
    # Se não houver conversão, o reconhecimento pode falhar porque as corres foram interpretadas
    # de maneira equivocada.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar as faces no frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Verificar cada face detectada
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Comparar a face detectada com as faces conhecidas
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # Definir o nome com base na comparação
        name = "Desconhecido"
        if True in matches:
            name = "Conhecido"

        # Definir a cor do retângulo com base no nome
        rectangle_color = (0, 255, 0)  # Verde para "Conhecido"
        if name == "Desconhecido":
            rectangle_color = (0, 0, 255)  # Vermelho para "Desconhecido"

        # Desenhar um retângulo ao redor da face e colocar um texto indicando o resultado
        cv2.rectangle(frame, (left, top), (right, bottom), rectangle_color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rectangle_color, 2)

    # Exibir o frame com as faces detectadas
    cv2.imshow('Video', frame)

    # Pressionar 's' para sair do programa
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# Liberar os recursos
video_capture.release()
cv2.destroyAllWindows()
