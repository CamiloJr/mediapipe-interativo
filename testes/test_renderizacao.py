import numpy as np
from motor_mediapipe import Config, Analise
from renderizacao import renderizar,relatorio,ETAPAS_EXPLICADAS

def p(x,y,z=0,vis=1):return (x,y,z,vis)

def analise():
    face=[p(.5,.5) for _ in range(478)]
    pose=[p(.5,.5) for _ in range(33)]
    pose[11]=p(.2,.2);pose[13]=p(.2,.4);pose[15]=p(.5,.4)
    hand=[p(.2,.2) for _ in range(21)]
    hand[0]=p(.2,.5)
    return Analise(np.zeros((120,180,3),np.uint8),'Holístico','imagem.png',25.0,
        rosto=[face],expressoes=[{'Sorriso':.83,'Boca aberta':.13}],
        corpo=[pose],mascaras=[np.ones((120,180),np.float32)],
        maos=[hand],lateralidade=['Right'],gestos=[('Thumb_Up',.91)])

def test_render_todos_estagios():
    a=analise()
    for etapa in range(5):
        frame=renderizar(a,Config(),etapa)
        assert frame.shape==(120,180,3)
        assert frame.dtype==np.uint8
    assert np.array_equal(renderizar(a,Config(),0),a.imagem)
    assert np.count_nonzero(renderizar(a,Config(),2))>0

def test_segmentacao_nao_altera_sem_mascara():
    a=Analise(np.zeros((120,180,3),np.uint8),'Corpo','x',1)
    assert renderizar(a,Config(),4).shape==(120,180,3)

def test_mascara_on_off():
    a=analise()
    cfg=Config()
    ativo=renderizar(a,cfg,4)
    inativo=renderizar(a,Config(mascara=False),4)
    assert not np.array_equal(ativo,inativo)

def test_relatorio_engloba_gesto_angulo_e_face():
    r=relatorio(analise(),Config(modo='Holístico'))
    txt=' '.join(k+':'+v for k,v in r)
    assert 'Sorriso' in txt
    assert 'Cotovelo esq.' in txt
    assert 'Polegar para cima' in txt
    assert '33 pontos' in txt

def test_paineis_didaticos():
    assert len(ETAPAS_EXPLICADAS)==5
    assert 'NÃO' in ETAPAS_EXPLICADAS['Interpretação'][2]
