"""Laboratório interativo da Aula 7 — MediaPipe e percepção visual multimodal.

Interface Pygame (estética e disposição inspiradas na classificação interativa
anterior), entrada por imagem e webcam, modelos reais via MediaPipe Tasks.
"""
from __future__ import annotations
import ctypes
from dataclasses import replace
from pathlib import Path
import sys
import cv2
import numpy as np
try:
    import pygame
except ImportError as exc:
    raise SystemExit('Instale as dependências: python -m pip install -r requirements.txt') from exc

from motor_mediapipe import (Config, MODOS, ETAPAS, abrir_imagem)
from renderizacao import renderizar, relatorio, ETAPAS_EXPLICADAS
from trabalhador import Trabalhador

W,H,FPS=1600,980,35
BG=(10,13,19); PANEL=(20,25,34); DARK=(13,17,25); BORDER=(55,66,81)
TEXT=(239,243,249); MUTED=(159,172,190); DIM=(105,124,144)
CYAN=(77,208,221); GREEN=(86,215,158); RED=(246,116,123); GOLD=(248,197,95)
BUTTON=(43,52,66); HOVER=(60,77,91)


def dpi_aware():
    if sys.platform=='win32':
        try: ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try: ctypes.windll.user32.SetProcessDPIAware()
            except Exception: pass


def cortar(fonte, msg, largura):
    msg=str(msg)
    if fonte.size(msg)[0]<=largura:
        return msg
    while msg and fonte.size(msg+'…')[0]>largura:
        msg=msg[:-1]
    return msg+'…'

class App:
    def __init__(self,size=None,worker=None):
        dpi_aware()
        pygame.init()
        pygame.display.set_caption('MediaPipe Interativo | Aula 7 - Inteligência Artificial')
        info=pygame.display.Info()
        if size is None:
            dw=info.current_w if info.current_w>=800 else W
            dh=info.current_h if info.current_h>=550 else H
            fator=min(1,dw*.96/W,dh*.92/H)
            size=(max(800,round(W*fator)),max(520,round(H*fator)))
        self.window=pygame.display.set_mode(size,pygame.RESIZABLE)
        self.canvas=pygame.Surface((W,H))
        self.clock=pygame.time.Clock()
        self.fonts={
            'title':pygame.font.SysFont('segoeui,arial',30,bold=True),
            'head':pygame.font.SysFont('segoeui,arial',20,bold=True),
            'bold':pygame.font.SysFont('segoeui,arial',17,bold=True),
            'body':pygame.font.SysFont('segoeui,arial',17),
            'small':pygame.font.SysFont('segoeui,arial',14),
            'tiny':pygame.font.SysFont('segoeui,arial',12),
            'mono':pygame.font.SysFont('consolas,monospace',15)}
        self.cfg=Config()
        self.worker=worker if worker is not None else Trabalhador()
        self.fonte='Nenhuma entrada selecionada'
        self.imagem=None
        self.resultado=None
        self.etapa=3
        self.revisao=-1
        self.pag_resultado=0
        self.status='Escolha um modo, carregue uma imagem ou ligue a webcam.'
        self.erro=False
        self.modal=False
        self.buttons={}
        self.viewport=(0,0,1.0)
        self.cache={}
        self.output_key=None
        self.output_img=None
        self.running=True

    def text(self,msg,x,y,font='body',color=TEXT,max_width=None):
        f=self.fonts[font]
        txt=cortar(f,msg,max_width) if max_width is not None else str(msg)
        self.canvas.blit(f.render(txt,True,color),(int(x),int(y)))

    def center(self,msg,rect,font='small',color=TEXT):
        r=pygame.Rect(rect)
        txt=cortar(self.fonts[font],str(msg),r.width-10)
        surf=self.fonts[font].render(txt,True,color)
        self.canvas.blit(surf,surf.get_rect(center=r.center))

    def panel(self,rect,bg=PANEL,border=BORDER):
        rect=pygame.Rect(rect)
        pygame.draw.rect(self.canvas,bg,rect,border_radius=10)
        pygame.draw.rect(self.canvas,border,rect,1,border_radius=10)

    def section(self,num,titulo,x,y):
        self.text(num,x,y,'head',CYAN)
        self.text(titulo,x+41,y,'head')

    def logical_mouse(self,pos):
        ox,oy,s=self.viewport
        return ((pos[0]-ox)/s,(pos[1]-oy)/s)

    def button(self,key,title,rect,callback,kind='normal',enabled=True,font='small'):
        rect=pygame.Rect(rect)
        hover=enabled and rect.collidepoint(self.logical_mouse(pygame.mouse.get_pos()))
        if not enabled: bg,fg=(30,38,48),(94,110,125)
        elif kind=='primary': bg,fg=((56,167,182) if hover else (32,141,155)),(255,255,255)
        elif kind=='active': bg,fg=(40,94,117),TEXT
        elif kind=='danger':bg,fg=(94,51,56),TEXT
        else:bg,fg=(HOVER if hover else BUTTON),TEXT
        pygame.draw.rect(self.canvas,bg,rect,border_radius=6)
        pygame.draw.rect(self.canvas,BORDER,rect,1,border_radius=6)
        self.center(title,rect,font,fg)
        self.buttons[key]=(rect,callback,enabled)

    def dialogo(self,salvar=False):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root=tk.Tk()
            root.withdraw()
            root.attributes('-topmost',True)
            try:
                if salvar:
                    return filedialog.asksaveasfilename(title='Exportar resultado MediaPipe',
                            defaultextension='.png',initialfile='mediapipe_resultado.png',
                            filetypes=[('Imagem PNG','*.png')])
                return filedialog.askopenfilename(title='Escolher imagem para MediaPipe',
                            filetypes=[('Imagens','*.jpg *.jpeg *.png *.webp *.bmp'),('Todos','*.*')])
            finally:
                root.destroy()
        except Exception as exc:
            self.status=f'Diálogo de arquivos indisponível: {exc}'
            self.erro=True
            return ''

    def abrir(self):
        caminho=self.dialogo()
        if not caminho:return
        try:
            img=abrir_imagem(caminho)
            self.worker.parar_camera()
            self.imagem=img
            self.resultado=None
            self.revisao=self.worker.snapshot().revisao
            self.fonte=Path(caminho).name
            self.pag_resultado=0
            self.cache.clear()
            self.status=f'{self.fonte} carregada. Clique em Analisar para inferência real.'
            self.erro=False
        except Exception as exc:
            self.status=f'Imagem inválida: {exc}'
            self.erro=True

    def analisar(self):
        if self.imagem is None:
            self.status='Carregue uma imagem ou use a webcam.'
            return
        if self.worker.snapshot().ativo:
            self.status='Pare a webcam antes de executar uma análise estática.'
            return
        self.resultado=None
        self.cache.clear()
        self.worker.analisar(self.imagem,self.fonte,self.cfg)
        self.status=f'Analisando {self.cfg.modo.lower()} com MediaPipe...'
        self.erro=False

    def camera(self):
        estado=self.worker.snapshot()
        if estado.ativo:
            self.worker.parar_camera()
            self.status='Captura interrompida. O último quadro permanece disponível.'
        else:
            self.resultado=None
            self.imagem=None
            self.fonte=f'Webcam {self.cfg.camera}'
            self.cache.clear()
            self.worker.iniciar_camera(self.cfg)
            self.status=f'Iniciando webcam {self.cfg.camera} / modo {self.cfg.modo}...'
            self.erro=False

    def congelar(self):
        estado=self.worker.snapshot()
        if estado.ativo:
            self.worker.parar_camera()
            if self.resultado is not None:
                self.imagem=self.resultado.imagem.copy()
                self.fonte='Quadro congelado'
            self.status='Quadro congelado: você pode mudar modos e clicar em Analisar.'
        elif self.resultado:
            self.imagem=self.resultado.imagem.copy()
            self.fonte='Quadro congelado'
            self.status='Quadro disponível para reanálise.'
        else:
            self.status='Não há quadro processado.'

    def exportar(self):
        if self.resultado is None:
            self.status='Não há inferência para exportar.'
            return
        caminho=self.dialogo(True)
        if not caminho:return
        try:
            result=renderizar(self.resultado,self.cfg,self.etapa)
            ok,buffer=cv2.imencode('.png',result)
            if not ok:raise OSError('Falha ao gerar PNG')
            buffer.tofile(caminho)
            self.status=f'Resultado exportado: {Path(caminho).name}'
        except Exception as exc:
            self.status=f'Erro ao exportar: {exc}'
            self.erro=True

    def modo(self,modo):
        if modo==self.cfg.modo:return
        self.cfg=replace(self.cfg,modo=modo)
        self.resultado=None
        self.pag_resultado=0
        self.cache.clear()
        self.worker.atualizar_cfg(self.cfg)
        self.status=(f'Modo {modo} selecionado. ' +
                     ('A webcam atualizará automaticamente.' if self.worker.snapshot().ativo
                      else 'Clique em Analisar para processar a imagem.'))

    def valor_config(self,chave,delta):
        cfg=self.cfg
        if chave in ('confianca','rastreamento','visibilidade'):
            cfg=replace(cfg,**{chave:round(min(.95,max(.10,getattr(cfg,chave)+delta*.05)),2)})
        elif chave=='opacidade':
            cfg=replace(cfg,opacidade=round(min(.9,max(.1,cfg.opacidade+delta*.1)),2))
        elif chave=='espessura':cfg=replace(cfg,espessura=max(1,min(5,cfg.espessura+delta)))
        elif chave=='max_pessoas':cfg=replace(cfg,max_pessoas=max(1,min(4,cfg.max_pessoas+delta)))
        elif chave=='max_maos':cfg=replace(cfg,max_maos=max(1,min(4,cfg.max_maos+delta)))
        elif chave=='camera':cfg=replace(cfg,camera=max(0,min(5,cfg.camera+delta)))
        elif chave=='densidade_face':
            valores=(1,2,3,4,6,8)
            i=valores.index(cfg.densidade_face)
            cfg=replace(cfg,densidade_face=valores[max(0,min(len(valores)-1,i+delta))])
        elif chave=='espelhar':cfg=replace(cfg,espelhar=not cfg.espelhar)
        elif chave=='mascara':cfg=replace(cfg,mascara=not cfg.mascara)
        cfg.validar()
        self.cfg=cfg
        self.cache.clear()
        self.worker.atualizar_cfg(cfg)
        # Alterações apenas de desenho têm efeito imediato; parâmetros da rede pedem reanálise.
        desenho={'opacidade','espessura','visibilidade','densidade_face','mascara'}
        self.status=('Visualização atualizada.' if chave in desenho else
                     ('Configuração aplicada aos próximos quadros.' if self.worker.snapshot().ativo else
                      'Parâmetro de inferência alterado. Clique em Analisar novamente.'))

    def atualizar(self):
        e=self.worker.snapshot()
        if e.revisao!=self.revisao and e.resultado is not None:
            self.revisao=e.revisao
            # Descartar predições produzidas para modo antigo após troca de painel.
            if e.resultado.modo==self.cfg.modo:
                self.resultado=e.resultado
                self.imagem=e.resultado.imagem
                self.fonte=e.resultado.origem
                self.cache.clear()
                self.pag_resultado=0
                self.status=e.status
                self.erro=False
        if e.erro and (self.status!=e.status):
            self.status=e.status
            self.erro=True

    def header(self):
        self.text('MediaPipe Interativo',27,14,'title')
        self.text('Visão computacional e compreensão de gestos, expressões e poses',28,53,'body',MUTED)
        self.text('AULA 7  •  MODELOS REAIS  •  MEDIA PIPE TASKS',1112,26,'small',CYAN)
        self.text('PIPELINE',28,83,'small',MUTED)
        self.text('Imagem → detecção → pontos 3D → conexões → interpretação',111,81,'bold')
        e=self.worker.snapshot()
        self.text('● WEBCAM' if e.ativo else '● PROCESSANDO' if e.ocupado else
                  '● RESULTADO' if self.resultado else '● PRONTO PARA COMEÇAR',
                  1017,84,'small', GREEN if e.ativo else GOLD if e.ocupado else CYAN if self.resultado else MUTED)
        for i,modo in enumerate(MODOS):
            self.button('modo'+str(i),f'{i+1}  {modo}',(27+i*312,113,302,38),
                        lambda m=modo:self.modo(m),'active' if self.cfg.modo==modo else 'normal',font='bold')
        pos=(27,289,551,813,1075,1337)
        labels=('Abrir imagem','Analisar','Parar webcam' if e.ativo else 'Iniciar webcam',
                'Congelar quadro','Exportar PNG','Como usar')
        calls=(self.abrir,self.analisar,self.camera,self.congelar,self.exportar,self.ajuda)
        for i,(lab,fun) in enumerate(zip(labels,calls)):
            enabled=(True,
                     self.imagem is not None and not e.ativo and not e.ocupado,
                     not e.ocupado or e.ativo,
                     e.ativo or self.resultado is not None,
                     self.resultado is not None,True)[i]
            self.button('acao'+str(i),lab,(pos[i],166,240,38),fun,
                        'primary' if i==1 else 'active' if i==2 and e.ativo else 'normal',enabled)

    def mostrar_imagem(self,img,rect,cache_key):
        rect=pygame.Rect(rect)
        self.panel(rect,DARK,(63,76,90))
        if img is None:
            self.center('Aguardando imagem...',rect,'body',MUTED)
            return
        box=rect.inflate(-10,-10)
        h,w=img.shape[:2]
        escala=min(box.w/w,box.h/h)
        sw,sh=max(1,round(w*escala)),max(1,round(h*escala))
        size=(sw,sh)
        destino=pygame.Rect(0,0,sw,sh)
        destino.center=box.center
        key=(cache_key,id(img),size)
        surf=self.cache.get(key)
        if surf is None:
            rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            tmp=pygame.image.frombuffer(rgb.tobytes(),(w,h),'RGB').copy()
            surf=pygame.transform.smoothscale(tmp,size)
            self.cache[key]=surf
            if len(self.cache)>10:
                self.cache.pop(next(iter(self.cache)))
        self.canvas.blit(surf,destino.topleft)

    def painel_visual(self):
        self.panel((24,220,1030,566))
        self.section('01','VISUALIZAÇÃO DO PROCESSAMENTO',44,238)
        if self.imagem is not None:
            h,w=self.imagem.shape[:2]
            self.text(f'{w}×{h} px',900,244,'small',MUTED)
        else:self.text('SEM ENTRADA',903,244,'small',MUTED)
        for i,nome in enumerate(ETAPAS):
            self.button('etapa'+str(i),f'{i+1}  {nome}',(42+i*198,277,190,38),
                        lambda k=i:self.mudar_etapa(k),
                        'active' if self.etapa==i else 'normal')
        self.text('IMAGEM DE ENTRADA',45,325,'small',MUTED)
        self.text(f'SAÍDA: {ETAPAS[self.etapa].upper()}',551,325,'small',CYAN)
        self.mostrar_imagem(self.imagem,(41,350,489,354),('entrada',self.revisao))
        if self.resultado is not None:
            chave=(self.revisao,self.cfg,self.etapa)
            if self.output_key!=chave:
                self.output_img=renderizar(self.resultado,self.cfg,self.etapa)
                self.output_key=chave
            saida=self.output_img
        else:
            saida=self.imagem
        self.mostrar_imagem(saida,(543,350,489,354),('saida',self.revisao,self.etapa,self.cfg))
        self.text('Origem: '+self.fonte,44,718,'small',MUTED,max_width=480)
        if self.resultado:
            self.text(f'Tempo: {self.resultado.ms:.0f} ms  |  modo: {self.resultado.modo}',549,718,'small',TEXT)
        else:
            self.text('Aguardando inferência para este modo.',549,718,'small',MUTED)
        pygame.draw.line(self.canvas,BORDER,(44,749),(1033,749),1)
        self.text('DICA',45,759,'small',CYAN)
        self.text('Use os cinco estágios para explicar o fluxo do algoritmo passo a passo.',101,756,'body')

    def mudar_etapa(self,k):
        self.etapa=k
        self.cache.clear()
        self.status=f'Etapa {k+1}: {ETAPAS[k]}. Veja a explicação no painel inferior.'

    def cfg_opcoes(self):
        c=[('Confiança mínima',f'{self.cfg.confianca:.2f}','confianca'),
           ('Conf. rastreamento',f'{self.cfg.rastreamento:.2f}','rastreamento'),
           ('Câmera / índice',str(self.cfg.camera),'camera'),
           ('Espelho na webcam','Ligado' if self.cfg.espelhar else 'Desligado','espelhar'),
           ('Espessura das linhas',str(self.cfg.espessura),'espessura'),
           ('Opacidade da máscara',f'{self.cfg.opacidade*100:.0f}%','opacidade')]
        if self.cfg.modo=='Rosto':
            c.extend([('Máximo de rostos',str(self.cfg.max_pessoas),'max_pessoas'),
                      ('Densidade dos pontos',str(self.cfg.densidade_face),'densidade_face')])
        elif self.cfg.modo=='Corpo':
            c.extend([('Máximo de pessoas',str(self.cfg.max_pessoas),'max_pessoas'),
                      ('Visibilidade mínima',f'{self.cfg.visibilidade:.2f}','visibilidade'),
                      ('Máscara corporal','Ligada' if self.cfg.mascara else 'Desligada','mascara')])
        elif self.cfg.modo=='Holístico':
            c.extend([('Máx. rostos / pessoas',str(self.cfg.max_pessoas),'max_pessoas'),
                      ('Máximo de mãos',str(self.cfg.max_maos),'max_maos'),
                      ('Máscara corporal','Ligada' if self.cfg.mascara else 'Desligada','mascara')])
        else:
            c.extend([('Máximo de mãos',str(self.cfg.max_maos),'max_maos')])
        return c

    def painel_config(self):
        self.panel((1070,220,506,367))
        self.section('02','CONFIGURAÇÕES',1090,237)
        self.text('Parâmetros de IA e visualização (CPU local).',1094,267,'tiny',MUTED)
        for i,(nome,valor,chave) in enumerate(self.cfg_opcoes()):
            y=283+i*32
            self.text(nome,1094,y+7,'small',MUTED,max_width=240)
            self.button('minus-'+chave,'−',(1344,y,34,29),lambda k=chave:self.valor_config(k,-1),font='bold')
            self.panel((1386,y,126,29),DARK,BORDER)
            self.center(valor,(1387,y,124,29),'small',CYAN)
            self.button('plus-'+chave,'+',(1517,y,34,29),lambda k=chave:self.valor_config(k,1),font='bold')

    def painel_resultado(self):
        self.panel((1070,600,506,330))
        self.section('03','RESULTADOS E MEDIDAS',1090,618)
        dados=relatorio(self.resultado,self.cfg)
        por_pagina=8
        total=max(1,(len(dados)+por_pagina-1)//por_pagina)
        self.pag_resultado=max(0,min(self.pag_resultado,total-1))
        if total>1:
            self.button('pag_ant','◀',(1354,620,36,28),lambda:self.mudar_pagina(-1),enabled=self.pag_resultado>0)
            self.text(f'{self.pag_resultado+1}/{total}',1411,626,'small',CYAN)
            self.button('pag_prox','▶',(1515,620,36,28),lambda:self.mudar_pagina(1),enabled=self.pag_resultado<total-1)
        for i,(chave,valor) in enumerate(dados[self.pag_resultado*por_pagina:(self.pag_resultado+1)*por_pagina]):
            y=666+i*28
            if i%2==0:
                pygame.draw.rect(self.canvas,(25,31,42),(1090,y-4,462,27),border_radius=4)
            self.text(chave,1097,y,'small',MUTED,max_width=202)
            self.text(valor,1302,y,'small',TEXT,max_width=247)
        if self.resultado is None:
            self.text('As previsões aparecerão aqui após a análise.',1097,825,'small',MUTED,max_width=430)
        else:
            self.text('Valores estimados por redes neurais (não são medições clínicas).',1097,902,'tiny',MUTED,max_width=447)

    def mudar_pagina(self,delta):
        self.pag_resultado+=delta

    def painel_didatico(self):
        self.panel((24,800,1030,131))
        nome=ETAPAS[self.etapa]
        titulo,l1,l2=ETAPAS_EXPLICADAS[nome]
        self.text(titulo,43,815,'bold',CYAN)
        self.text(l1,44,851,'body',TEXT,max_width=990)
        self.text(l2,44,890,'small',MUTED,max_width=987)

    def rodape(self):
        pygame.draw.line(self.canvas,BORDER,(25,947),(1574,947),1)
        self.text('STATUS',28,958,'small',RED if self.erro else CYAN)
        self.text(self.status,92,956,'small',TEXT,max_width=1465)

    def ajuda(self):
        self.modal=True

    def draw_help(self):
        shade=pygame.Surface((W,H),pygame.SRCALPHA)
        shade.fill((0,0,0,196))
        self.canvas.blit(shade,(0,0))
        self.panel((202,147,1196,666),(24,31,43),(85,107,129))
        self.text('Guia didático — MediaPipe e interpretação visual',234,174,'head',CYAN)
        linhas=[
            '1. Escolha Rosto, Corpo, Mãos, Gestos ou Holístico na faixa superior.',
            '2. Abra uma foto JPG/PNG ou inicie a webcam. Não existem exemplos pré-carregados.',
            '3. Nas fotos, clique Analisar. Na webcam, a IA processa quadros continuamente.',
            '4. Alterne Original → Pontos → Conexões → Interpretação → Segmentação.',
            '5. Rosto: 478 landmarks e 52 coeficientes faciais (alguns aparecem no painel).',
            '6. Corpo: 33 landmarks, ângulos projetados 2D e máscara da pessoa.',
            '7. Mãos: 21 landmarks por mão e lateralidade; Gestos: sete categorias comuns.',
            '8. Holístico: executa os três modelos, com maior custo computacional.',
            '9. Configurações de inferência na imagem fixa exigem clicar Analisar novamente.',
            '10. Congele a webcam para revisar uma imagem; exporte a etapa atual em PNG.',
            '11. Em imagens de pessoas, respeite a autorização e a privacidade dos retratados.',
            'ATENÇÃO: não há reconhecimento confiável de emoções internas a partir do rosto.',
            'ATENÇÃO: os gestos pré-definidos NÃO constituem tradução de Libras.',
            'TECLAS: 1–5 etapas; Tab muda modo; Espaço analisa; S exporta; H ajuda; Esc fecha.'
        ]
        for i,linha in enumerate(linhas):
            cor=GOLD if linha.startswith('ATENÇÃO') else TEXT
            self.text(linha,237,230+i*37,'small',cor,max_width=1103)
        self.button('help-close','Fechar  (Esc)',(1091,749,240,43),lambda:setattr(self,'modal',False),'primary')

    def desenhar(self):
        self.buttons.clear()
        self.canvas.fill(BG)
        self.header()
        self.painel_visual()
        self.painel_config()
        self.painel_resultado()
        self.painel_didatico()
        self.rodape()
        if self.modal:
            self.buttons.clear()
            self.draw_help()
        ww,hh=self.window.get_size()
        escala=min(ww/W,hh/H)
        nw,nh=round(W*escala),round(H*escala)
        ox,oy=(ww-nw)//2,(hh-nh)//2
        self.viewport=(ox,oy,escala)
        self.window.fill(BG)
        if nw==W and nh==H:self.window.blit(self.canvas,(ox,oy))
        else:self.window.blit(pygame.transform.smoothscale(self.canvas,(nw,nh)),(ox,oy))
        pygame.display.flip()

    def evento(self,evt):
        if evt.type==pygame.QUIT:self.running=False
        elif evt.type==pygame.VIDEORESIZE:
            self.window=pygame.display.set_mode((max(800,evt.w),max(520,evt.h)),pygame.RESIZABLE)
        elif evt.type==pygame.KEYDOWN:
            if evt.key==pygame.K_ESCAPE:
                if self.modal:self.modal=False
                else:self.running=False
            elif not self.modal:
                if pygame.K_1<=evt.key<=pygame.K_5:self.mudar_etapa(evt.key-pygame.K_1)
                elif evt.key==pygame.K_TAB:
                    self.modo(MODOS[(MODOS.index(self.cfg.modo)+1)%len(MODOS)])
                elif evt.key==pygame.K_SPACE:self.analisar()
                elif evt.key==pygame.K_s:self.exportar()
                elif evt.key==pygame.K_h:self.ajuda()
        elif evt.type==pygame.MOUSEBUTTONDOWN and evt.button==1:
            xy=self.logical_mouse(evt.pos)
            for rect,callback,enabled in list(self.buttons.values()):
                if enabled and rect.collidepoint(xy):
                    callback()
                    return

    def executar(self):
        try:
            while self.running:
                for evt in pygame.event.get():self.evento(evt)
                self.atualizar()
                self.desenhar()
                self.clock.tick(FPS)
        finally:
            self.worker.fechar()
            pygame.quit()


def main():
    App().executar()

if __name__=='__main__':
    main()
