"""
Ponto de entrada:  python3 -m visualizador [--amostrar N] [--sem-janela]

Roda a MESMA simulação de machine_learning.py e abre uma janela flutuante
mostrando, ao vivo, o código em execução e a rede neural sendo treinada.
"""

import argparse

from .simulacao import main


def _argumentos():
    p = argparse.ArgumentParser(
        prog="python3 -m visualizador",
        description="Aprendizado federado (HMEQ) com visualização ao vivo da rede neural.",
    )
    p.add_argument("--amostrar", type=int, default=None, metavar="N",
                   help="animar 1 a cada N lotes (padrão: 1 lote por época)")
    p.add_argument("--sem-janela", action="store_true",
                   help="roda só no terminal (útil para comparar com o original)")
    return p.parse_args()


if __name__ == "__main__":
    args = _argumentos()
    main(amostrar_a_cada=args.amostrar, janela=not args.sem_janela)
