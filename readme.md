## Sobre o uso de face_recognition

A biblioteca face_recognition usa uma abordagem diferente para o reconhecimento facial
e não possui as mesmas funções de treinamento direto como o OpenCV. Em vez disso,
ela usa embeddings faciais que são criados para cada face detectada.
Embeddings faciais são representações vetoriais (ou numéricas) das características de uma face.
Esses vetores são gerados por algoritmos de aprendizado de máquina e redes neurais
para capturar as características únicas e discriminativas das faces,
permitindo a comparação e reconhecimento entre diferentes rostos.


Instalar dependências

    pip install matplotlib
    pip install numpy
    pip install scipy
    pip install mkl

    pip install opencv-python (se necessário, pip install opencv-python-headless)

    pip install face_recognition 
    (https://stackoverflow.com/questions/56696940/how-to-install-face-recognition-module-for-python)

Se ocorrer problema na instalação das dependências

    pip install cmake    
    pip install wheel
    pip install dlib (https://stackoverflow.com/questions/52829483/importerror-no-module-named-dlib)
    pip install --upgrade pip setuptools

    Outras instalações: https://github.com/ageitgey/face_recognition/issues/175#issue-257710508

Dependências de compilação (que podem ser necessárias)

    sudo apt-get update
    sudo apt-get install build-essential cmake
    sudo apt-get install libgtk-3-dev
    sudo apt-get install libboost-all-dev

Instalar dlib pelo anaconda (opcional)

    conda install -c conda-forge dlib

Para ativar venv se precisar

    python -m venv reconhecimento_facial
