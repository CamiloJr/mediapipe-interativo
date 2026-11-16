"""Desenho explicável dos landmarks, conexões, máscaras e rótulos previstos."""
from __future__ import annotations
import cv2
import numpy as np
from motor_mediapipe import Analise, Config, GESTOS_PT, angulos_corpo

POSE_CONNECTIONS = [
 (11,12),(11,13),(13,15),(12,14),(14,16),
 (11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),
 (27,29),(29,31),(28,30),(30,32),(15,17),(15,19),(15,21),
 (16,18),(16,20),(16,22),(0,1),(1,2),(2,3),(3,7),
 (0,4),(4,5),(5,6),(6,8),(9,10)
]
HAND_CONNECTIONS = [
 (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
 (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
 (13,17),(17,18),(18,19),(19,20),(0,17)
]
FACE_CONTOURS = [
 (10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10),
 (33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33),
 (263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466,263),
 (61,40,37,0,267,270,291,321,314,17,84,91,61),
]
CORES = {'rosto': (255, 176, 81), 'corpo': (83, 222, 132), 'maos': (231, 157, 67),
         'texto': (242, 247, 249)}


def _xy(p, w, h):
    return (int(np.clip(p[0] * w, 0, w - 1)), int(np.clip(p[1] * h, 0, h - 1)))


def _pontos_e_linhas(img, lista, arestas, cor, config, desenhar_linhas, salto=1, visibilidade=False):
    h,w = img.shape[:2]
    if desenhar_linhas:
        for a,b in arestas:
            if a < len(lista) and b < len(lista):
                if visibilidade and min(lista[a][3], lista[b][3]) < config.visibilidade:
                    continue
                cv2.line(img,_xy(lista[a],w,h),_xy(lista[b],w,h),cor,config.espessura,cv2.LINE_AA)
    for i,p in enumerate(lista):
        if i % salto == 0 and (not visibilidade or p[3]>=config.visibilidade):
            cv2.circle(img,_xy(p,w,h),max(1,config.espessura+1),cor,-1,cv2.LINE_AA)


def _texto(img, txt, x, y, cor=(255,255,255)):
    h,w=img.shape[:2]
    x=int(np.clip(x,0,max(0,w-10)))
    y=int(np.clip(y,20,max(20,h-10)))
    escala = min(.75,max(.46,w/1300))
    (tw,th), baseline=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,escala,1)
    cv2.rectangle(img,(x,max(0,y-th-8)),(min(w-1,x+tw+12),min(h-1,y+5)),(20,29,39),-1)
    cv2.putText(img,txt,(x+5,y),cv2.FONT_HERSHEY_SIMPLEX,escala,cor,1,cv2.LINE_AA)


def _segmentar(img, analise: Analise, cfg: Config):
    if not analise.mascaras:
        return
    h,w=img.shape[:2]
    original = img.copy()
    cor=np.empty_like(img);cor[:]=(76,184,232)
    mask_total=np.zeros((h,w),dtype=np.float32)
    for mask in analise.mascaras:
        if mask.shape[:2] != (h,w):
            mask=cv2.resize(mask,(w,h),interpolation=cv2.INTER_LINEAR)
        mask_total=np.maximum(mask_total,mask)
    mask_total=np.clip(mask_total*cfg.opacidade,0,1)[...,None]
    img[:]=(original.astype(np.float32)*(1-mask_total) + cor.astype(np.float32)*mask_total).astype(np.uint8)


def renderizar(analise: Analise | None, cfg: Config, etapa: int = 3) -> np.ndarray | None:
    if analise is None:
        return None
    img=analise.imagem.copy()
    if etapa==0:
        return img
    desenhar_linhas=etapa>=2
    mostrar_texto=etapa>=3
    if etapa==4 and cfg.mascara:
        _segmentar(img,analise,cfg)
    for i,face in enumerate(analise.rosto):
        h,w=img.shape[:2]
        if etapa not in (4,) or not analise.mascaras:
            _pontos_e_linhas(img,face,[],CORES['rosto'],cfg,False,
                            salto=cfg.densidade_face if etapa != 1 else max(1,cfg.densidade_face*2))
            if desenhar_linhas:
                for contorno in FACE_CONTOURS:
                    _pontos_e_linhas(img,face,list(zip(contorno,contorno[1:])),CORES['rosto'],cfg,True,salto=9999)
        if mostrar_texto:
            x=int(min(p[0] for p in face)*w) if face else 5
            y=int(min(p[1] for p in face)*h) if face else 23
            _texto(img,f'Face #{i+1}  478 pontos',x,y,CORES['rosto'])
    for i,pose in enumerate(analise.corpo):
        _pontos_e_linhas(img,pose,POSE_CONNECTIONS,CORES['corpo'],cfg,desenhar_linhas,
                        visibilidade=True)
        if mostrar_texto:
            x,y=_xy(pose[11],img.shape[1],img.shape[0]) if len(pose)>11 else (15,50)
            _texto(img,f'Corpo #{i+1}  33 pontos',x,y-25,CORES['corpo'])
            if len(pose)>16:
                ang=angulos_corpo(pose,cfg.visibilidade)
                if 'Cotovelo esq.' in ang:
                    p=_xy(pose[13],img.shape[1],img.shape[0])
                    _texto(img,f'{ang["Cotovelo esq."]:.0f} graus',p[0]+12,p[1]-10,CORES['corpo'])
    for i,mao in enumerate(analise.maos):
        _pontos_e_linhas(img,mao,HAND_CONNECTIONS,CORES['maos'],cfg,desenhar_linhas)
        if mostrar_texto and mao:
            x,y=_xy(mao[0],img.shape[1],img.shape[0])
            lado=analise.lateralidade[i] if i<len(analise.lateralidade) else '?'
            label='Dir.' if lado=='Right' else 'Esq.' if lado=='Left' else '?'
            gesto=analise.gestos[i] if i <len(analise.gestos) else ('None',0)
            txt=f'Mao {label} 21 pontos'
            if cfg.modo in ('Gestos','Holístico'):
                # Caixa em ASCII para renderização OpenCV; descrição completa fica na interface.
                txt=f'{label}: {gesto[0]} ({gesto[1]*100:.0f}%)'
            _texto(img,txt,x-14,y+25,CORES['maos'])
    if etapa==4 and not cfg.mascara:
        _texto(img,'Mascara desativada nas configuracoes',12,img.shape[0]-16)
    elif etapa==4 and not analise.mascaras:
        _texto(img,'Mascara disponivel no modo Corpo / Holistico',12,img.shape[0]-16)
    return img


def relatorio(analise: Analise | None, cfg: Config) -> list[tuple[str,str]]:
    if analise is None:
        return [('Aguardando', 'Abra uma imagem ou inicie a webcam.')]
    linhas=[]
    c=analise.contagens
    linhas.append(('Detecção',f'{c["rostos"]} rosto(s) | {c["corpos"]} corpo(s) | {c["mãos"]} mão(s)'))
    if analise.rosto:
        linhas.append(('Malha facial',f'{len(analise.rosto[0])} pontos 3D normalizados'))
        if analise.expressoes:
            for nome,score in sorted(analise.expressoes[0].items(),key=lambda e:-e[1]):
                linhas.append((nome, f'{score:.3f} (coef. geométrico)'))
        else:
            linhas.append(('Expressões', 'Coeficientes não disponibilizados pelo modelo'))
    if analise.corpo:
        linhas.append(('Esqueleto',f'{len(analise.corpo[0])} pontos por pessoa'))
        for nome,valor in angulos_corpo(analise.corpo[0],cfg.visibilidade).items():
            linhas.append((nome,f'{valor:.1f}° (projeção 2D)'))
        linhas.append(('Máscara', 'Disponível' if analise.mascaras else 'Não produzida'))
    for i,mao in enumerate(analise.maos):
        lado=analise.lateralidade[i] if i<len(analise.lateralidade) else '?'
        pinca = None
        from motor_mediapipe import distancia_pinca
        pinca=distancia_pinca(mao)
        linhas.append((f'Mão {i+1}', f'{lado}: {len(mao)} pontos; pinça={pinca if pinca is not None else "?"}'))
        if cfg.modo in ('Gestos','Holístico') and i<len(analise.gestos):
            classe,score=analise.gestos[i]
            linhas.append((f'Gesto {i+1}',f'{GESTOS_PT.get(classe,classe)} ({score*100:.1f}%)'))
    linhas.append(('Inferência',f'{analise.ms:.0f} ms por quadro'))
    return linhas


ETAPAS_EXPLICADAS = {
 'Original': ('01 • AQUISIÇÃO',
              'Entrada por fotografia ou quadro da webcam. Imagem RGB fornecida ao MediaPipe.',
              'Não há treinamento; os pesos da rede já foram aprendidos anteriormente.'),
 'Pontos': ('02 • LOCALIZAÇÃO DE PONTOS-CHAVE',
            'A IA estima coordenadas 2D/3D: até 478 na face, 33 no corpo e 21 por mão.',
            'Cada ponto representa uma região anatômica; não equivale ao reconhecimento de intenção.'),
 'Conexões': ('03 • ESTRUTURA GEOMÉTRICA',
             'Conectamos pontos relevantes para reconstruir esqueleto e contornos faciais.',
             'As linhas são desenhadas para visualização; a rede prevê os pontos.'),
 'Interpretação': ('04 • INTERPRETAÇÃO DA INFERÊNCIA',
                  'Gestos: classes pré-treinadas. Corpo: ângulos 2D. Rosto: coeficientes de expressão.',
                  'Expressão facial NÃO prova emoção; gestos pré-treinados NÃO traduzem Libras.'),
 'Segmentação': ('05 • SEGMENTAÇÃO DE PESSOAS',
                 'O Pose Landmarker produz uma máscara de confiança por pixel da pessoa.',
                 'A máscara está disponível nos modos Corpo e Holístico; não é segmentação de objetos.'),
}
