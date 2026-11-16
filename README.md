# MediaPipe Interativo — Aula 7 | Introdução à Inteligência Artificial

Software didático em **Python + Pygame Community Edition + MediaPipe Tasks + OpenCV** para observar, comparar e explicar, em tempo real, aplicações de redes neurais a imagens humanas. Desenvolvido com foco na apresentação em sala de aula e seguindo o padrão visual do projeto anterior *Visão Computacional Interativa*.

**Uso:** imagens selecionadas do computador (JPG, PNG, BMP, WebP) ou câmera ao vivo (webcam). **Não há imagens de exemplo nem botões de exemplo.**

> **Precisão conceitual:** identificação de rostos, pose, mãos e gestos por imagem é **visão computacional**, não processamento de linguagem natural (PLN) em sentido estrito. Pode integrar sistemas multimodais de interpretação de linguagem, mas não constitui, isoladamente, PLN.

## Funcionalidades

| Modo | Modelo oficial MediaPipe | O que demonstra |
|---|---|---|
| **Rosto** | Face Landmarker | Localização de até 478 marcos faciais 3D e coeficientes de expressões faciais (blendshapes) |
| **Corpo** | Pose Landmarker (Lite) | 33 pontos da pose, conexões esqueléticas, ângulos **estimados na projeção 2D** e máscara de segmentação da pessoa |
| **Mãos** | Gesture Recognizer (landmarks) | 21 pontos por mão, lateralidade estimada e geometria dos dedos, incluindo distância normalizada entre polegar e indicador |
| **Gestos** | Gesture Recognizer | Classificação de sete gestos comuns, com pontuação do classificador, além de landmarks |
| **Holístico** | Face + Pose + Gesture | Combina as três inferências para visualizar rosto, corpo e mãos em uma cena; requer mais processamento |

A tarefa *Gesture Recognizer* identifica `Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory` e `ILoveYou`; quando não identifica nenhuma dessas classes, retorna `None`. O sinal `ILoveYou` é uma categoria de **ASL**; **o software NÃO é tradutor de Libras** nem reconhece linguagem de sinais em geral.

### Sobre "emoções"

O MediaPipe Face Landmarker não fornece diagnóstico psicológico nem uma classificação validada da emoção vivida pela pessoa. A aplicação exibe **indicadores geométricos e coeficientes de expressão observáveis**: sorriso, abertura da boca, fechamento dos olhos, elevação/contração de sobrancelhas etc. Uma pontuação `Sorriso = 0,80` **não representa 80% de probabilidade de felicidade**. Interpretações como "raiva", "tristeza" ou "felicidade" dependem de contexto e não podem ser inferidas com confiabilidade apenas do rosto.

## Execução no Windows (recomendado)

**Requisitos:** Windows 10/11 64 bits, Python **3.12, 3.13 ou 3.14 de 64 bits**, webcam opcional, internet na primeira instalação e no primeiro uso de cada modelo.

1. Extraia o arquivo ZIP inteiro para uma pasta gravável, por exemplo `Documentos\MediaPipe_Interativo_Aula_7`.
2. Abra `INICIAR_WINDOWS.bat` com duplo clique. O arquivo cria a `.venv`, instala as bibliotecas Python e inicia a aplicação.
3. Escolha **Rosto, Corpo, Mãos, Gestos ou Holístico**.
4. Clique em **Abrir imagem**, selecione uma foto e use **Analisar**; ou clique em **Iniciar webcam**.
5. Alterne as cinco etapas para acompanhar os resultados. No primeiro uso de um modo, os pesos oficiais serão baixados para `modelos/`.

Para executar manualmente no terminal (PowerShell, dentro da pasta):

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe mediapipe_interativo.py
```

**Nota sobre o Python 3.14:** a partir do MediaPipe 1.2.3 há distribuição para Windows de 64 bits compatível com Python 3.14; `pygame-ce` também oferece suporte. Esta edição do iniciador aceita Python 3.14. Evite instalar `pygame` clássico: a dependência correta é **`pygame-ce`**. Caso já exista uma pasta `.venv` criada com outra versão do Python, renomeie-a para `.venv_antiga` antes de iniciar para criar um ambiente novo.

### Executando no Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python mediapipe_interativo.py
```

No Linux, uma câmera física deve estar acessível ao OpenCV; em ambientes sem interface gráfica, a interface não pode ser aberta.

## Fluxo didático na interface

O programa mantém a **imagem original à esquerda** e uma **saída à direita**, com cinco visualizações:

1. **Original:** aquisição da foto ou captura de quadro da webcam.
2. **Pontos:** landmarks estimados pela rede (x, y, z normalizados).
3. **Conexões:** estrutura reconstruída sobre os landmarks: dedos, esqueleto, olhos e contorno facial.
4. **Interpretação:** sinais de expressões, rótulos de gestos e ângulos geométricos.
5. **Segmentação:** máscara por pixel da pessoa, quando o modo **Corpo** ou **Holístico** produzir esse resultado; não é segmentação de classes de objetos.

A área direita traz controles de confiança mínima, confiança de rastreamento, câmera, espelhamento, espessura, opacidade, quantidade de mãos/pessoas, visibilidade/densidade de landmarks e ativação da máscara quando aplicável.

**Importante:** em fotografias, mudanças que afetam a rede neural (confiança ou número máximo de pessoas/mãos) exigem clicar **Analisar** novamente. Alterações puramente visuais são exibidas imediatamente. Na webcam, os parâmetros passam a valer nos quadros seguintes.

### Controles úteis

| Controle | Ação |
|---|---|
| `1` a `5` | Selecionar uma das cinco etapas visuais |
| `Tab` | Alternar modo de análise |
| `Espaço` | Analisar a imagem selecionada |
| `S` | Exportar visualização atual para PNG |
| `H` | Ajuda integrada |
| `Esc` | Fechar ajuda ou sair |
| `Congelar quadro` | Pausar webcam e preservar um quadro para inspeção/reanálise |

## Conceitos para discutir com a turma

**Detecção vs. pontos:** uma rede localiza uma pessoa/face/mão; outra estima a estrutura de pontos relevantes. **Marcos geométricos** não são automaticamente rótulos semânticos de emoções, ações ou intenções.

**Reconhecimento de gestos:** o modelo transforma a geometria da mão em classes pré-treinadas. Uma mão pode ser localizada corretamente, mas o gesto ser classificado como `None`. Essa é uma ótima demonstração de diferença entre **localização** e **classificação**.

**Estimativa de pose:** o software calcula ângulos do cotovelo e joelho usando os pontos **na imagem**. Isso fornece uma *projeção 2D*, que muda com a perspectiva da câmera; não representa a medida biomecânica exata de uma articulação no espaço 3D.

**Segmentação:** o Pose Landmarker gera uma **máscara de confiança de primeiro plano** para a pessoa. Os pontos correspondem a posições geométricas; a máscara representa probabilidades espacializadas de pertencer à pessoa.

**Holístico:** nesta implementação é a execução conjunta de **três tarefas MediaPipe independentes**, e não o antigo pacote *MediaPipe Holistic* com um único grafo. Essa distinção ajuda a explicar arquiteturas multimodais modulares.

## Modelos — origem oficial e transferência

No primeiro uso, são baixados três arquivos **oficiais** disponibilizados pelo Google:

- Face: [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
- Corpo: [pose_landmarker_lite.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task)
- Mãos e gestos: [gesture_recognizer.task](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task)

Os arquivos `.task` ficam em `modelos/` e são reutilizados. O software usa **inferência local**: as imagens e os quadros **não são transmitidos para processamento em nuvem pelo aplicativo**. O download inicial dos pesos e das dependências exige conexão. Por isso, **prepare os modelos antes de levar o computador à sala de aula sem internet**. Também é possível colocar manualmente os três `.task` na pasta `modelos/`, preservando os nomes acima.

Fontes oficiais: [Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker?tab=Python), [Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker?tab=Python), [Gesture Recognizer](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer?tab=Python), [guia de configuração](https://developers.google.com/edge/mediapipe/solutions/setup_python).

## Estrutura do projeto

```text
MediaPipe_Interativo_Aula_7/
├── mediapipe_interativo.py  # Interface Pygame, modos, controles e painéis
├── motor_mediapipe.py      # MediaPipe Tasks, download/execução dos modelos
├── renderizacao.py         # Pontos, conexões, máscaras e explicações
├── trabalhador.py          # Thread para câmera e inferência (UI responsiva)
├── modelos/
│   └── LEIA-ME.txt          # Os arquivos .task são baixados no primeiro uso
├── testes/
│   ├── test_motor.py
│   ├── test_renderizacao.py
│   └── test_tarefas_simuladas.py
├── requirements.txt
├── INICIAR_WINDOWS.bat
└── README.md
```

## Testes e limites

Para executar os testes unitários:

```bash
python -m pytest -q
```

Os testes validam geometria, tratamento de scores, saídas, renderização de máscaras, modos de análise e integração simulada com a API. **Não verificam o funcionamento dos pesos neurais reais nem da webcam física.** O programa foi preparado com base na documentação oficial, mas, neste ambiente de produção, não foi possível instalar MediaPipe/Pygame para um ensaio completo. Recomenda-se testar os cinco modos localmente antes da demonstração pública.

### Solução de problemas

### Correção de compatibilidade com campos opcionais (edição 1.1)

Se você recebeu `Falha do MediaPipe: float() argument must be a string or a real number, not 'NoneType'` ao iniciar a webcam, **esta edição corrige o problema**. O MediaPipe pode retornar campos opcionais `z`, `visibility` ou `score` com valor `None`. Agora `z` assume 0.0 quando ausente, `visibility` assume 1.0 quando ausente (somente para desenho), e scores opcionais assumem 0.0. Esses valores de substituição **não significam confiança medida pelo modelo**. A mesma correção vale para webcam e imagens.

Não é necessário reinstalar o Python: basta substituir os arquivos antigos pelos desta versão e executar `INICIAR_WINDOWS.bat`. A `.venv` existente pode ser reutilizada.

Em caso de novo erro de inferência, a aplicação escreve a exceção completa em `mediapipe_erros.log`, na pasta do programa, sem registrar imagens ou quadros. Esse arquivo pode ser consultado para diagnóstico.

- **Erro ao instalar pygame:** use Python 3.12–3.14 e `pygame-ce`; não instale `pygame` clássico.
- **Não abre webcam:** feche aplicativos que possam estar usando a câmera e altere o índice de 0 para 1 (ou superior).
- **Modelo não baixa:** verifique acesso a `storage.googleapis.com`, proxy/firewall e permissões da pasta `modelos/`. Você pode baixar os `.task` manualmente pelos links oficiais.
- **Lento no modo Holístico:** selecione um único modo, reduza a resolução enviada pela câmera ou use um processador mais rápido.
- **Sem pontos na imagem:** melhore iluminação, enquadramento e nitidez; reduza com cuidado o limiar de confiança.
- **Mão invertida:** o modo espelho altera a relação esquerda/direita na imagem; desative-o para evitar inversões aparentes.

## Uso responsável

O software é voltado para **ensino, demonstração e experimentação**, não para decisões clínicas, avaliação psicológica, vigilância, identificação de pessoas ou tradução profissional de línguas de sinais. Peça consentimento antes de usar a webcam com participantes. Não há armazenamento automático dos quadros; apenas a função **Exportar PNG** grava uma imagem quando acionada.

**Descrição curta sugerida para GitHub:** *Software didático em Python com MediaPipe e Pygame para análise interativa de expressões faciais, pose corporal, mãos e gestos, por imagem ou webcam, com visualização explicável das etapas da IA.*
