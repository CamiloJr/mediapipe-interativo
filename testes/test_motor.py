from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
import pytest
from motor_mediapipe import (Config, MODOS, angulo_2d, angulos_corpo,
    sinais_faciais, distancia_pinca, Analise, Motor)

def p(x=.5,y=.5,z=0,vis=1):return (x,y,z,vis)

def test_config_validacao():
    Config().validar()
    with pytest.raises(ValueError):replace(Config(), modo='modelo falso').validar()
    with pytest.raises(ValueError):replace(Config(), confianca=1.4).validar()
    with pytest.raises(ValueError):replace(Config(), max_maos=0).validar()

@pytest.mark.parametrize('modo', MODOS)
def test_mododidatico(modo):
    replace(Config(),modo=modo).validar()

def test_angulo_direito_e_alinhamento():
    assert angulo_2d(p(0,1),p(0,0),p(1,0))==90.0
    assert angulo_2d(p(0,0),p(1,0),p(2,0))==180.0
    assert angulo_2d(p(0,0),p(0,0),p(0,0)) is None

def test_angulo_visibilidade_pose():
    arr=[p() for _ in range(33)]
    arr[11]=p(.1,.2);arr[13]=p(.1,.4);arr[15]=p(.3,.4)
    assert angulos_corpo(arr)['Cotovelo esq.']==90.0
    arr[13]=p(.1,.4,0,.2)
    assert 'Cotovelo esq.' not in angulos_corpo(arr,.5)

def test_sinais_nao_inferem_emocao():
    cs=[NS(category_name='mouthSmileLeft',score=.72),
        NS(category_name='mouthSmileRight',score=.40),
        NS(category_name='jawOpen',score=.2)]
    result=sinais_faciais(cs)
    assert result['Sorriso']==.72
    assert result['Boca aberta']==.2
    assert 'Felicidade' not in result

def test_distancia_pinca_normalizada():
    arr=[p() for _ in range(21)]
    arr[0]=p(0,0);arr[9]=p(0,1);arr[4]=p(0,0);arr[8]=p(0,.2)
    assert distancia_pinca(arr)==.2
    assert distancia_pinca(arr[:5]) is None

def test_contagens():
    a=Analise(np.zeros((100,100,3),dtype=np.uint8),'Rosto','teste',9.,rosto=[[p()]],maos=[[p()]])
    assert a.contagens=={'rostos':1,'corpos':0,'mãos':1}
