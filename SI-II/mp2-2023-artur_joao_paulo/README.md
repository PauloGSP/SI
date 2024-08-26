

# SI2 - Mini-projecto nº 2
Este mini-projecto está centrado na matéria de aprendizagem por reforço, tendo por aplicação o jogo "Bomberman".

Na disciplina de Inteligência Artificial (IA), seguindo o enunciado "bomberman.pdf", os alunos desenvolveram agentes para jogar este jogo autonomamente. Foram fornecidos o motor do jogo ("server.py") e um visualizador ("viewer.py") e os alunos apresentaram agentes, um dos quais é agora fornecido a título exemplificativo ("student.py").

Em IA, estes agentes eram tipicamente pré-programados, ou seja, não usavam mecanismos da área da aprendizagem automática.

Agora, neste mini-projecto de SI2, os alunos devem desenvolver agentes que, recorrendo a aprendizagem por reforço, consigam adquirir a funcionalidade necessária para jogar o jogo do Bomberman.

![Demo](https://github.com/dgomes/iia-ia-bomberman/raw/master/data/DemoBomberman.gif)

## How to install

Make sure you are running Python 3.5 or higher.

`$ pip install -r requirements.txt`

*Tip: you might want to create a virtualenv first*

## How to play

open 3 terminals:

`$ python3 server.py`

`$ python3 viewer.py`

`$ python3 client.py`

to play using the sample client make sure the client pygame hidden window has focus

### Keys

Directions: arrows

*A*: 'a' - detonates (only after picking up the detonator powerup)

*B*: 'b' - drops bomb

## Debug Installation

Make sure pygame is properly installed:

python -m pygame.examples.aliens

# Tested on:
- Ubuntu 18.04
- OSX 10.14.6
- Windows 10.0.18362

