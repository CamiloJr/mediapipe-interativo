"""Testa a integração com a API das tarefas por objetos simulados (sem pesos reais)."""
from types import SimpleNamespace as NS
import sys
import numpy as np
from motor_mediapipe import Motor, Config

class ImagemFake:
    def __init__(self,**kwargs):self.kwargs=kwargs
class ModeloFake:
    def __init__(self,classe):self.classe=classe
    def detect(self,img):return self.prever()
    def detect_for_video(self,img,ms):
        assert ms>0
        return self.prever()
    def recognize(self,img):return self.prever()
    def recognize_for_video(self,img,ms):
        assert ms>0
        return self.prever()
    def prever(self):
        p=NS(x=.2,y=.5,z=.01,visibility=.9)
        if self.classe=='rosto':
            return NS(face_landmarks=[[p]*478], face_blendshapes=[[
                NS(category_name='mouthSmileLeft',score=.65)]])
        if self.classe=='corpo':
            return NS(pose_landmarks=[[p]*33],segmentation_masks=[
                NS(numpy_view=lambda:np.ones((80,100),np.float32))])
        return NS(hand_landmarks=[[p]*21], handedness=[[
            NS(category_name='Right')]], gestures=[[
            NS(category_name='Victory',score=.94)]])

def test_integracao_5_modos(monkeypatch):
    fake=NS(Image=ImagemFake,ImageFormat=NS(SRGB='srgb'))
    monkeypatch.setitem(sys.modules,'mediapipe',fake)
    motor=Motor()
    monkeypatch.setattr(motor,'_pegar',lambda chave,cfg,video:ModeloFake(chave))
    entrada=np.zeros((80,100,3),np.uint8)
    for modo in ('Rosto','Corpo','Mãos','Gestos','Holístico'):
        from dataclasses import replace
        r=motor.processar(entrada,replace(Config(),modo=modo),'teste',video=True)
        assert r.imagem.shape==(80,100,3)
        assert (len(r.rosto)>0)==(modo in ('Rosto','Holístico'))
        assert (len(r.corpo)>0)==(modo in ('Corpo','Holístico'))
        assert (len(r.maos)>0)==(modo in ('Mãos','Gestos','Holístico'))
        if r.maos:
            assert r.gestos[0]==('Victory',.94)
        if r.rosto:
            assert r.expressoes[0]['Sorriso']==.65
        if r.corpo:
            assert len(r.mascaras)==1
