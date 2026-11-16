"""Regressão: algumas versões da API retornam z/visibility/score=None."""
from types import SimpleNamespace as NS
import sys
import numpy as np
import pytest

from motor_mediapipe import _pontos, sinais_faciais, Motor, Config
from renderizacao import relatorio


def test_landmarks_com_campos_none_e_ausentes():
    pontos = [NS(x=.2, y=.4, z=None, visibility=None),
              NS(x=.3, y=.5),
              NS(x=.4, y=.6, z=0., visibility=0.)]
    assert _pontos(pontos) == [(.2, .4, 0., 1.), (.3, .5, 0., 1.), (.4, .6, 0., 0.)]


def test_landmark_sem_coordenadas_obrigatorias_mostra_erro_compreensivel():
    with pytest.raises(ValueError, match='x/y ausentes'):
        _pontos([NS(x=None, y=.3)])


def test_score_opcional_da_expressao():
    saida = sinais_faciais([NS(category_name='mouthSmileLeft', score=None),
                           NS(category_name='jawOpen', score=.55)])
    assert saida['Sorriso'] == 0.0
    assert saida['Boca aberta'] == .55


class ImagemFake:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class ModeloFake:
    def __init__(self, chave):
        self.chave = chave

    def detect_for_video(self, img, ts):
        assert ts > 0
        pt = NS(x=.3, y=.5, z=None, visibility=None)
        if self.chave == 'rosto':
            return NS(face_landmarks=[[pt] * 478], face_blendshapes=[])
        return NS(pose_landmarks=[[pt] * 33], segmentation_masks=[])

    def recognize_for_video(self, img, ts):
        pt = NS(x=.4, y=.4, z=None, visibility=None)
        return NS(hand_landmarks=[[pt] * 21],
                  handedness=[[NS(category_name='Right')]],
                  gestures=[[NS(category_name='Victory', score=None)]])


def test_todos_modos_em_video_sem_float_none(monkeypatch):
    monkeypatch.setitem(sys.modules, 'mediapipe', NS(Image=ImagemFake, ImageFormat=NS(SRGB='srgb')))
    motor = Motor()
    monkeypatch.setattr(motor, '_pegar', lambda chave, cfg, video: ModeloFake(chave))
    quadro = np.zeros((80, 100, 3), dtype=np.uint8)
    for modo in ('Rosto', 'Corpo', 'Mãos', 'Gestos', 'Holístico'):
        resultado = motor.processar(quadro, Config(modo=modo), 'Webcam 0', video=True)
        assert resultado.modo == modo
        assert resultado.gestos == ([('Victory', 0.0)] if modo in ('Mãos', 'Gestos', 'Holístico') else [])
        if modo in ('Rosto', 'Holístico'):
            assert resultado.rosto[0][0] == (.3, .5, 0., 1.)
            assert ('Expressões', 'Coeficientes não disponibilizados pelo modelo') in relatorio(resultado, Config(modo=modo))


def test_diagnostico_em_arquivo_sem_gravar_imagens(monkeypatch, tmp_path):
    import trabalhador
    destino = tmp_path / 'log.txt'
    monkeypatch.setattr(trabalhador, 'ARQUIVO_DIAGNOSTICO', destino)
    try:
        raise TypeError("float() argument must be a string or a real number, not 'NoneType'")
    except TypeError as exc:
        trabalhador.registrar_erro(exc, 'webcam', Config(modo='Gestos'))
    txt = destino.read_text(encoding='utf-8')
    assert 'Modo=Gestos' in txt
    assert 'TypeError' in txt
    assert 'Traceback' in txt
    assert 'float() argument' in txt
