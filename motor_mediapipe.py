"""Motor didático: tarefas MediaPipe (MediaPipe Tasks API).

O aplicativo executa inferência real com modelos pré-treinados; não treina modelos.
Rosto -> FaceLandmarker; Corpo -> PoseLandmarker; Mãos/Gestos -> GestureRecognizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import math
import os
import time
import urllib.request

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELOS = ROOT / 'modelos'
MODOS = ('Rosto', 'Corpo', 'Mãos', 'Gestos', 'Holístico')
ETAPAS = ('Original', 'Pontos', 'Conexões', 'Interpretação', 'Segmentação')
URL_MODELOS = {
    'rosto': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
    'corpo': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
    'gestos': 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task',
}
NOMES_MODELOS = {k: url.rsplit('/', 1)[-1] for k, url in URL_MODELOS.items()}
GESTOS_PT = {
    'Closed_Fist': 'Punho fechado', 'Open_Palm': 'Palma aberta',
    'Pointing_Up': 'Indicador para cima', 'Thumb_Down': 'Polegar para baixo',
    'Thumb_Up': 'Polegar para cima', 'Victory': 'Vitória (V)',
    'ILoveYou': 'Sinal "I love you" (ASL)', 'None': 'Não reconhecido',
    'Unknown': 'Não reconhecido',
}
EXPRESSOES = (
    ('Sorriso', ('mouthSmileLeft', 'mouthSmileRight')),
    ('Boca aberta', ('jawOpen',)),
    ('Olhos fechados', ('eyeBlinkLeft', 'eyeBlinkRight')),
    ('Sobrancelhas erguidas', ('browInnerUp', 'browOuterUpLeft', 'browOuterUpRight')),
    ('Sobrancelhas franzidas', ('browDownLeft', 'browDownRight')),
    ('Boca contraída', ('mouthPucker',)),
)

@dataclass(frozen=True)
class Config:
    modo: str = 'Rosto'
    confianca: float = .50
    rastreamento: float = .50
    visibilidade: float = .50
    max_pessoas: int = 1
    max_maos: int = 2
    opacidade: float = .43
    espessura: int = 2
    camera: int = 0
    espelhar: bool = True
    mascara: bool = True
    densidade_face: int = 3

    def validar(self) -> None:
        if self.modo not in MODOS:
            raise ValueError(f'Modo inválido: {self.modo}')
        for chave in ('confianca', 'rastreamento', 'visibilidade', 'opacidade'):
            valor = getattr(self, chave)
            if not 0.0 <= valor <= 1.0:
                raise ValueError(f'{chave} fora de [0, 1]')
        if not 1 <= self.max_pessoas <= 4 or not 1 <= self.max_maos <= 4:
            raise ValueError('Número de pessoas/mãos fora da faixa')

@dataclass
class Analise:
    imagem: np.ndarray  # BGR, H x W x 3
    modo: str
    origem: str
    ms: float
    rosto: list[list[Any]] = field(default_factory=list)
    expressoes: list[dict[str, float]] = field(default_factory=list)
    corpo: list[list[Any]] = field(default_factory=list)
    mascaras: list[np.ndarray] = field(default_factory=list)
    maos: list[list[Any]] = field(default_factory=list)
    lateralidade: list[str] = field(default_factory=list)
    gestos: list[tuple[str, float]] = field(default_factory=list)

    @property
    def contagens(self) -> dict[str, int]:
        return {'rostos': len(self.rosto), 'corpos': len(self.corpo), 'mãos': len(self.maos)}


def abrir_imagem(caminho: str) -> np.ndarray:
    """Lê caminhos com acentos no Windows sem depender do cv2.imread."""
    dados = np.fromfile(caminho, dtype=np.uint8)
    img = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Formato de imagem inválido ou não suportado.')
    if img.shape[0] > 4096 or img.shape[1] > 4096:
        fator = min(4096 / img.shape[0], 4096 / img.shape[1])
        img = cv2.resize(img, None, fx=fator, fy=fator, interpolation=cv2.INTER_AREA)
    return img


def garantir_modelo(chave: str, informar: Callable[[str], None] | None = None) -> Path:
    MODELOS.mkdir(exist_ok=True, parents=True)
    caminho = MODELOS / NOMES_MODELOS[chave]
    if caminho.exists() and caminho.stat().st_size > 100_000:
        return caminho
    tmp = caminho.with_name(caminho.name + '.parcial')
    if informar:
        informar(f'Baixando modelo oficial do MediaPipe ({chave})...')
    try:
        req = urllib.request.Request(URL_MODELOS[chave], headers={'User-Agent': 'Mozilla/5.0 MediaPipe-Aula7/1.0'})
        with urllib.request.urlopen(req, timeout=90) as resposta, open(tmp, 'wb') as arq:
            while True:
                bloco = resposta.read(1024 * 512)
                if not bloco:
                    break
                arq.write(bloco)
        if tmp.stat().st_size < 100_000:
            raise OSError('Arquivo de modelo incompleto.')
        os.replace(tmp, caminho)
        return caminho
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f'Não foi possível obter {chave}. Verifique internet/rede; '
                           f'URL: {URL_MODELOS[chave]} | {exc}') from exc


def _numero_opcional(valor: Any, padrao: float) -> float:
    """Converte apenas números válidos; campos opcionais podem vir como None.

    No MediaPipe, visibility e z nem sempre são informados para rosto/mãos.
    Preservamos 0.0 quando esse é realmente o valor retornado pelo modelo.
    """
    if valor is None:
        return padrao
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return padrao
    return numero if math.isfinite(numero) else padrao


def _pontos(lista: Any) -> list[tuple[float, float, float, float]]:
    """Extrai landmarks sem tentar executar float(None).

    x/y são obrigatórios; z e visibility são opcionais. Quando a API não
    fornece visibility, assume-se 1.0 apenas para permitir exibição dos pontos.
    Não é uma estimativa de confiança produzida pelo modelo.
    """
    saida = []
    for indice, ponto in enumerate(lista):
        x, y = getattr(ponto, 'x', None), getattr(ponto, 'y', None)
        if x is None or y is None:
            raise ValueError(f'Landmark {indice}: coordenadas x/y ausentes.')
        x, y = float(x), float(y)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f'Landmark {indice}: coordenadas x/y não finitas.')
        z = _numero_opcional(getattr(ponto, 'z', None), 0.0)
        visibilidade = _numero_opcional(getattr(ponto, 'visibility', None), 1.0)
        saida.append((x, y, z, visibilidade))
    return saida


def sinais_faciais(classificacoes: Any) -> dict[str, float]:
    """Coeficientes geométricos *não* são probabilidades de emoções reais."""
    scores = {c.category_name: _numero_opcional(getattr(c, 'score', None), 0.0)
              for c in classificacoes if getattr(c, 'category_name', None)}
    return {rotulo: round(max((scores.get(chave, 0.0) for chave in chaves), default=0.0), 3)
            for rotulo, chaves in EXPRESSOES}


def angulo_2d(a: tuple, b: tuple, c: tuple) -> float | None:
    """Ângulo 2D no ponto b; não é o ângulo anatômico em 3D."""
    v = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    w = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
    den = np.linalg.norm(v) * np.linalg.norm(w)
    if den < 1e-9:
        return None
    return round(math.degrees(math.acos(float(np.clip(np.dot(v, w) / den, -1, 1)))), 1)


def angulos_corpo(pontos: list[tuple], minimo: float = .5) -> dict[str, float]:
    pares = {'Cotovelo esq.': (11, 13, 15), 'Cotovelo dir.': (12, 14, 16),
             'Joelho esq.': (23, 25, 27), 'Joelho dir.': (24, 26, 28)}
    saida = {}
    for nome, ids in pares.items():
        if len(pontos) <= max(ids) or any(pontos[i][3] < minimo for i in ids):
            continue
        valor = angulo_2d(*(pontos[i] for i in ids))
        if valor is not None:
            saida[nome] = valor
    return saida


def distancia_pinca(pontos: list[tuple]) -> float | None:
    """Distância polegar-indicador normalizada pelo tamanho aparente da palma."""
    if len(pontos) < 21:
        return None
    def dist(i, j):
        return math.hypot(pontos[i][0] - pontos[j][0], pontos[i][1] - pontos[j][1])
    escala = max(1e-6, dist(0, 9))
    return round(dist(4, 8) / escala, 2)


class Motor:
    def __init__(self, informar: Callable[[str], None] | None = None):
        self._tarefas: dict[tuple, Any] = {}
        self.informar = informar

    def _construir(self, chave: str, cfg: Config, video: bool):
        import mediapipe as mp
        caminho = garantir_modelo(chave, self.informar)
        visao = mp.tasks.vision
        base = mp.tasks.BaseOptions(model_asset_path=str(caminho))
        modo = visao.RunningMode.VIDEO if video else visao.RunningMode.IMAGE
        if chave == 'rosto':
            opts = visao.FaceLandmarkerOptions(
                base_options=base, running_mode=modo,
                num_faces=cfg.max_pessoas,
                min_face_detection_confidence=cfg.confianca,
                min_face_presence_confidence=cfg.confianca,
                min_tracking_confidence=cfg.rastreamento,
                output_face_blendshapes=True)
            return visao.FaceLandmarker.create_from_options(opts)
        if chave == 'corpo':
            opts = visao.PoseLandmarkerOptions(
                base_options=base, running_mode=modo,
                num_poses=cfg.max_pessoas,
                min_pose_detection_confidence=cfg.confianca,
                min_pose_presence_confidence=cfg.confianca,
                min_tracking_confidence=cfg.rastreamento,
                output_segmentation_masks=True)
            return visao.PoseLandmarker.create_from_options(opts)
        opts = visao.GestureRecognizerOptions(
            base_options=base, running_mode=modo, num_hands=cfg.max_maos,
            min_hand_detection_confidence=cfg.confianca,
            min_hand_presence_confidence=cfg.confianca,
            min_tracking_confidence=cfg.rastreamento)
        return visao.GestureRecognizer.create_from_options(opts)

    def _pegar(self, chave: str, cfg: Config, video: bool):
        key = (chave, video, cfg.confianca, cfg.rastreamento,
               cfg.max_pessoas if chave in ('rosto', 'corpo') else cfg.max_maos)
        if key not in self._tarefas:
            # Impede acúmulo de modelos com parâmetros antigos.
            for antigo in list(self._tarefas):
                if antigo[0] == chave and antigo != key:
                    self._tarefas.pop(antigo).close()
            if self.informar:
                self.informar(f'Inicializando IA: {chave} / {"vídeo" if video else "imagem"}')
            self._tarefas[key] = self._construir(chave, cfg, video)
        return self._tarefas[key]

    def processar(self, quadro: np.ndarray, cfg: Config, origem: str,
                  video: bool = False) -> Analise:
        cfg.validar()
        if quadro is None or quadro.ndim != 3 or quadro.shape[2] != 3:
            raise ValueError('Imagem BGR inválida')
        import mediapipe as mp
        inicio = time.perf_counter()
        # Reduz quadros muito grandes para manter o uso em sala de aula fluido.
        h, w = quadro.shape[:2]
        fator = min(1.0, 1024 / max(h, w))
        if fator < 1:
            quadro = cv2.resize(quadro, (round(w * fator), round(h * fator)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)
        imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts = int(time.monotonic() * 1000)
        resposta = Analise(imagem=quadro.copy(), modo=cfg.modo, origem=origem, ms=0)
        chaves = {'Rosto': ('rosto',), 'Corpo': ('corpo',),
                  'Mãos': ('gestos',), 'Gestos': ('gestos',),
                  'Holístico': ('rosto', 'corpo', 'gestos')}[cfg.modo]
        for chave in chaves:
            task = self._pegar(chave, cfg, video)
            if chave == 'rosto':
                resultado = task.detect_for_video(imagem_mp, ts) if video else task.detect(imagem_mp)
                resposta.rosto = [_pontos(lista) for lista in resultado.face_landmarks]
                resposta.expressoes = [sinais_faciais(lista) for lista in (resultado.face_blendshapes or [])]
            elif chave == 'corpo':
                resultado = task.detect_for_video(imagem_mp, ts) if video else task.detect(imagem_mp)
                resposta.corpo = [_pontos(lista) for lista in resultado.pose_landmarks]
                if resultado.segmentation_masks:
                    resposta.mascaras = [np.array(m.numpy_view(), dtype=np.float32).copy()
                                         for m in resultado.segmentation_masks]
            else:
                resultado = task.recognize_for_video(imagem_mp, ts) if video else task.recognize(imagem_mp)
                resposta.maos = [_pontos(lista) for lista in resultado.hand_landmarks]
                resposta.lateralidade = [lista[0].category_name if lista else 'Indefinida'
                                         for lista in resultado.handedness]
                resposta.gestos = [
                    (lista[0].category_name or 'None',
                     _numero_opcional(getattr(lista[0], 'score', None), 0.0))
                    if lista else ('None', 0.0)
                    for lista in resultado.gestures
                ]
        resposta.ms = (time.perf_counter() - inicio) * 1000
        return resposta

    def fechar(self):
        for tarefa in self._tarefas.values():
            try:
                tarefa.close()
            except Exception:
                pass
        self._tarefas.clear()
