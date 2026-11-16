"""Thread de captura/inferência. A UI nunca bloqueia por leitura da câmera ou IA."""
from __future__ import annotations
from dataclasses import dataclass
from threading import Lock, Event, Thread
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

from motor_mediapipe import Analise, Config, Motor

ARQUIVO_DIAGNOSTICO = Path(__file__).resolve().parent / 'mediapipe_erros.log'


def registrar_erro(exc: Exception, etapa: str, cfg: Config) -> None:
    """Guarda o traceback sem salvar imagem, vídeo ou qualquer dado biométrico."""
    try:
        with ARQUIVO_DIAGNOSTICO.open('a', encoding='utf-8') as arq:
            arq.write(f'\n[{datetime.now().isoformat(timespec="seconds")}] '
                      f'{etapa} | Modo={cfg.modo} | Camera={cfg.camera}\n')
            arq.write(f'Python: {sys.version.split()[0]} | Erro: {type(exc).__name__}: {exc}\n')
            arq.write(traceback.format_exc() + '\n')
    except OSError:
        pass  # Falhas ao salvar o log não impedem o status de erro na UI.


@dataclass(frozen=True)
class Estado:
    resultado: Analise | None
    revisao: int
    ativo: bool
    ocupado: bool
    status: str
    erro: bool

class Trabalhador:
    def __init__(self):
        self._lock = Lock()
        self._stop = Event()
        self._cfg = Config()
        self._geracao = 0
        self._pendente: tuple | None = None
        self._camera = False
        self._atual = Estado(None, 0, False, False,
                             'Selecione uma imagem ou inicie a webcam.',False)
        self._t = Thread(target=self._run, name='MediaPipe-Inferencia', daemon=True)
        self._t.start()

    def snapshot(self) -> Estado:
        with self._lock:
            return self._atual

    def _publicar(self, *, gen=None, **kwargs):
        with self._lock:
            if gen is not None and gen != self._geracao:
                return
            data=self._atual.__dict__.copy()
            data.update(kwargs)
            self._atual=Estado(**data)

    def _aviso(self, msg):
        self._publicar(status=msg, erro=False)

    def analisar(self, imagem: np.ndarray, origem: str, cfg: Config):
        with self._lock:
            self._cfg=cfg
            self._geracao+=1
            self._camera=False
            self._pendente=(imagem.copy(),origem,cfg,self._geracao)
            self._atual=Estado(self._atual.resultado,self._atual.revisao,
                               False,True,'Análise agendada. Inicializando modelo, se necessário...',False)

    def iniciar_camera(self,cfg:Config):
        with self._lock:
            self._cfg=cfg
            self._geracao+=1
            self._camera=True
            self._pendente=None
            self._atual=Estado(self._atual.resultado,self._atual.revisao,
                               True,True,f'Abrindo webcam {cfg.camera}...',False)

    def parar_camera(self):
        with self._lock:
            self._camera=False
            self._geracao+=1
            self._pendente=None
            self._atual=Estado(self._atual.resultado,self._atual.revisao,
                               False,False,'Webcam pausada. Último resultado preservado.',False)

    def atualizar_cfg(self,cfg:Config):
        with self._lock:
            anterior=self._cfg
            self._cfg=cfg
            if self._camera and cfg.camera!=anterior.camera:
                self._geracao+=1
            if self._camera:
                self._atual=Estado(self._atual.resultado,self._atual.revisao,
                     True,self._atual.ocupado,'Parâmetros aplicados aos próximos quadros.',False)

    def _run(self):
        cam=None
        cam_index=None
        motor=Motor(informar=self._aviso)
        ultimo_quadro=0.0
        try:
            while not self._stop.is_set():
                with self._lock:
                    pendente=self._pendente
                    if pendente:
                        self._pendente=None
                    camera=self._camera
                    cfg=self._cfg
                    gen=self._geracao
                if pendente is not None:
                    frame,origem,config,job=pendente
                    if cam is not None:
                        cam.release();cam=None;cam_index=None
                    try:
                        self._publicar(gen=job, ocupado=True,status='Executando modelos neurais...')
                        result=motor.processar(frame,config,origem,video=False)
                        with self._lock:
                            if job==self._geracao:
                                self._atual=Estado(result,self._atual.revisao+1,False,False,
                                    f'Inferência concluída: {result.ms:.0f} ms.',False)
                    except Exception as exc:
                        registrar_erro(exc, 'imagem', config)
                        self._publicar(gen=job, ocupado=False,
                                      status=f'Falha na análise: {type(exc).__name__}: {exc} '
                                             '(detalhes: mediapipe_erros.log)', erro=True)
                    continue
                if not camera:
                    if cam is not None:
                        cam.release();cam=None;cam_index=None
                    time.sleep(.05)
                    continue
                if cam is None or cam_index != cfg.camera:
                    if cam is not None:
                        cam.release()
                    backend=cv2.CAP_DSHOW if sys.platform=='win32' else cv2.CAP_ANY
                    cam=cv2.VideoCapture(cfg.camera,backend)
                    if not cam.isOpened():
                        cam.release();cam=None
                        with self._lock:
                            if gen==self._geracao:
                                self._camera=False
                                self._atual=Estado(self._atual.resultado,self._atual.revisao,False,False,
                                                   f'Não foi possível abrir a webcam {cfg.camera}.',True)
                        continue
                    cam_index=cfg.camera
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH,960)
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,540)
                    cam.set(cv2.CAP_PROP_BUFFERSIZE,1)
                ok,frame=cam.read()
                if not ok or frame is None:
                    with self._lock:
                        if gen==self._geracao:
                            self._camera=False
                            self._atual=Estado(self._atual.resultado,self._atual.revisao,False,False,
                                               'A webcam parou de enviar quadros.',True)
                    continue
                if cfg.espelhar:
                    frame=cv2.flip(frame,1)
                try:
                    result=motor.processar(frame,cfg,f'Webcam {cfg.camera}',video=True)
                    with self._lock:
                        if gen==self._geracao and self._camera:
                            self._atual=Estado(result,self._atual.revisao+1,True,False,
                                 f'Webcam • {result.ms:.0f} ms/modelos • {result.imagem.shape[1]}×{result.imagem.shape[0]}',False)
                except Exception as exc:
                    registrar_erro(exc, 'webcam', cfg)
                    with self._lock:
                        if gen==self._geracao:
                            self._camera=False
                            self._atual=Estado(self._atual.resultado,self._atual.revisao,
                                               False,False,
                                               f'Falha do MediaPipe: {type(exc).__name__}: {exc} '
                                               '(detalhes: mediapipe_erros.log)', True)
                # Limite suave; evita saturar CPU na demonstração.
                intervalo=time.monotonic()-ultimo_quadro
                if intervalo<.065:
                    time.sleep(.065-intervalo)
                ultimo_quadro=time.monotonic()
        finally:
            if cam is not None:
                cam.release()
            motor.fechar()

    def fechar(self):
        self._stop.set()
        with self._lock:
            self._camera=False
        self._t.join(timeout=2.0)
