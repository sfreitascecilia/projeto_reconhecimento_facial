import face_recognition
import cv2
import os
import pickle
import sys


# Função para capturar as imagens usadas no treinamento
def capture_faces(img_folder, registration):
    # Configurações iniciais
    camera = cv2.VideoCapture(0)  # 0 para a câmera padrão
    amostra = 1
    numero_amostras = 25  # número de amostras a serem capturadas

    print(f"\nSerão capturadas 25 amostras da face")
    print(f"Para a captura, pressione a tecla 'c'")
    print(f"e acompanhe o resultado da captura no terminal")

    id_faces = input(f"\nSelecione o identificador de faces (1 - sem máscara / 2 - com máscara): ")

    largura, altura = 200, 200  # dimensões das imagens capturadas

    print("Capturando as faces...")

    while True:
        connected, image_cam = camera.read()  # captura as imagens da webcam
        if not connected:
            print("Falha ao capturar a face!")
            break

        gray_image = cv2.cvtColor(image_cam, cv2.COLOR_RGB2GRAY)  # converte para cinza

        # Encontra todas as faces na image_cam
        face_locations = face_recognition.face_locations(gray_image)

        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(image_cam, (left, top), (right, bottom), (0, 0, 255), 2)  # desenha retângulo em volta da face

            if cv2.waitKey(1) & 0xFF == ord('c'):  # teclar 'c' para realizar a captura das imagens
                face_image = gray_image[top:bottom, left:right]  # recortar a face
                face_image_resized = cv2.resize(face_image, (largura, altura))  # redimensionar a face

                # Salvar a imagem
                file_path = f"{img_folder}/employee.{registration}.{id_faces}.{amostra}.jpg"

                # Minha percepção é a de que a imagem fica mais nítida quando voltamos para o formato RGB
                # mas, caso não queira utilizar a conversão para este formato, comentar a linha abaixo e
                # retirar o comentário da linha posterior
                cv2.imwrite(file_path, cv2.cvtColor(face_image_resized, cv2.COLOR_RGB2BGR))  # salva no formato BGR
                # cv2.imwrite(file_path, face_image_resized)

                print(
                    f"[Face {amostra} capturada com sucesso. "
                    f"Restam {numero_amostras - amostra} amostras para capturar]")
                amostra += 1

        cv2.imshow("Face", image_cam)  # mostra a imagem com retângulos desenhados
        if amostra > numero_amostras:
            break

    print("Faces capturadas com sucesso")
    camera.release()
    cv2.destroyAllWindows()


# Função para obter embeddings das faces
def get_face_encodings_from_images(img_folder):
    print("Obtendo representações das características da face...")
    face_encod = []
    ids_face = []

    # Percorrer todas as subpastas e/ou arquivos dentro da pasta 'images'
    for root, dirs, files in os.walk(img_folder):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                file_path = os.path.join(root, file)
                # Carregar a imagem e converter para RGB
                image = face_recognition.load_image_file(file_path)
                # Encontrar todas as faces na imagem
                face_locations = face_recognition.face_locations(image)
                # Extrair os embeddings das faces
                face_encodings_in_image = face_recognition.face_encodings(image, face_locations)

                if face_encodings_in_image:
                    # Adicionar todos os embeddings encontrados na lista
                    for encoding in face_encodings_in_image:
                        face_encod.append(encoding)
                        # Extraí o ID a partir do nome do arquivo
                        file_name = os.path.split(file_path)[-1]
                        face_id = int(file_name.split('.')[1])
                        ids_face.append(face_id)

    print("Representações da face obtidas com sucesso")
    return ids_face, face_encod


# Função para treinamento e salvamento dos embeddings
def train_and_save_encoders(ids_face, face_encod):
    print("Treinando...")

    # Criar a pasta 'embedding' se não existir
    encoding_folder = 'embedding'
    os.makedirs(encoding_folder, exist_ok=True)

    # Usar pickle para salvar embeddings e IDs
    with open(os.path.join(encoding_folder, 'encodings.pkl'), 'wb') as f:
        pickle.dump({'ids': ids_face, 'encodings': face_encod}, f)

    print("Treinamento realizado")


# Função para carregar os embeddings faciais
def load_encodings():
    encoding_folder = 'embedding'
    with open(os.path.join(encoding_folder, 'encodings.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data['ids'], data['encodings']


# Função para comparar os embeddings das novas imagens
def recognize_faces(image_path, known_face_id, known_face_encoding):
    # Carregar a imagem a ser reconhecida
    image = face_recognition.load_image_file(image_path)
    # Encontrar todas as faces na imagem
    face_locations = face_recognition.face_locations(image)
    # Obter os embeddings das faces encontradas
    face_encod = face_recognition.face_encodings(image, face_locations)

    face_ids = []

    for encoding in face_encod:
        # Comparar os embeddings da imagem com os embeddings conhecidos
        matches = face_recognition.compare_faces(known_face_encoding, encoding)
        distances = face_recognition.face_distance(known_face_encoding, encoding)
        best_match_index = distances.argmin()

        if matches[best_match_index]:
            face_ids.append(known_face_id[best_match_index])

    return face_ids


# Pasta onde as imagens estão localizadas
images_folder = 'images'
os.makedirs(images_folder, exist_ok=True)

print("*** Captura de Faces ***")

# Identificador de funcionários
employee = input(f"\nDigite a matrícula do funcionário: ")

if employee:
    # Captura as faces
    capture_faces(images_folder, employee)

    # Obtém embeddings faciais
    ids, face_encodings = get_face_encodings_from_images(images_folder)

    # Realiza o treinamento e salva os embeddings
    train_and_save_encoders(ids, face_encodings)

    # Carrega os encodings e identificadores salvos
    known_face_ids, known_face_encodings = load_encodings()

    # Loop para reconhecer várias imagens
    for cont in range(1, 3):  # Ajustar conforme total de imagens que se deseja reconhecer

        # Definir o caminho da imagem a ser reconhecida
        test_image_path = f'rosto_conhecido_{cont}.jpg'

        # Realiza o reconhecimento facial
        recognized_ids = recognize_faces(test_image_path, known_face_ids, known_face_encodings)

        # Imprime as matrículas reconhecidas
        if not recognized_ids:  # Verifica se a lista está vazia
            print(f"Não existem matrículas reconhecidas para a imagem '{test_image_path}'.")
        else:
            print(f"Matrícula reconhecida para a imagem '{test_image_path}':", recognized_ids)

else:
    print("Nenhuma matrícula foi digitada. O programa será encerrado.")
    sys.exit()
