# SIMULAÇÃO FEDERADA COM VISUALIZAÇÃO AO VIVO
#=========================================================
# Este módulo é um ESPELHO da função main() de machine_learning.py.
# Ele reaproveita tudo o que já existe lá, sem alterar uma linha:
#   carregar_e_preprocessar_dados, RedeNeural, treinar, avaliar,
#   ClienteHMEQ, media_ponderada e as variáveis globais (NUM_ROUNDS, ...).
#
# O que muda:
#   - ClienteVisual: herda de ClienteHMEQ e só "liga a câmera" (hooks)
#     durante o fit(). O treino em si é o treinar() original.
#   - FedAvgVisual: herda do FedAvg do Flower e avisa a janela quando o
#     servidor distribui, recebe e agrega os pesos.
#=========================================================

import os
import select
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from flwr.common import parameters_to_ndarrays

import machine_learning as ml   # o código original, importado como módulo

from .observador import ObservadorTreino
from .protocolo import Emissor, arredondar

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODIGO_ORIGINAL = os.path.join(RAIZ, "machine_learning.py")


# SEÇÃO 1 - CLIENTE OBSERVADO
#=========================================================
class ClienteVisual(ml.ClienteHMEQ):
    """ClienteHMEQ original + observador de hooks durante o treino local."""

    def __init__(self, loader_treino, loader_teste, input_dim, cid: int,
                 emissor: Emissor, amostrar_a_cada: Optional[int]):
        super().__init__(loader_treino, loader_teste, input_dim)
        self.cid = cid
        self.emissor = emissor
        self.amostrar_a_cada = amostrar_a_cada

    def fit(self, parametros, config: Dict):
        rodada = int(config.get("rodada", 0))
        lotes = len(self.loader_treino)
        self.emissor.enviar("recebeu", cliente=self.cid, rodada=rodada)
        observador = ObservadorTreino(
            self.modelo, self.emissor, self.cid, rodada,
            lotes_por_epoca=lotes,
            amostrar_a_cada=self.amostrar_a_cada or lotes,  # padrão: 1 lote por época
        )
        with observador:
            resultado = super().fit(parametros, config)   # -> treinar() original
        self.emissor.enviar("envio", cliente=self.cid, rodada=rodada, amostras=resultado[1])
        return resultado


# SEÇÃO 2 - ESTRATÉGIA FEDAVG OBSERVADA
#=========================================================
def _pesos(ndarrays: List[np.ndarray]) -> list:
    """Só as matrizes de pesos (ndim == 2), na ordem fc1, fc2, fc3."""
    return [arredondar(a.tolist()) for a in ndarrays if a.ndim == 2]


class FedAvgVisual(fl.server.strategy.FedAvg):
    def __init__(self, *args, emissor: Emissor, **kwargs):
        super().__init__(*args, **kwargs)
        self.emissor = emissor

    def configure_fit(self, server_round, parameters, client_manager):
        instrucoes = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in instrucoes:
            fit_ins.config["rodada"] = server_round   # o cliente fica sabendo o round
        self.emissor.enviar("rodada", rodada=server_round,
                            pesos=_pesos(parameters_to_ndarrays(parameters)))
        return instrucoes

    def aggregate_fit(self, server_round, results, failures):
        parametros, metricas = super().aggregate_fit(server_round, results, failures)
        if parametros is not None:
            self.emissor.enviar(
                "agregado", rodada=server_round,
                pesos=_pesos(parameters_to_ndarrays(parametros)),
                amostras=[fit_res.num_examples for _, fit_res in results],
            )
        return parametros, metricas

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metricas = super().aggregate_evaluate(server_round, results, failures)
        self.emissor.enviar("avaliacao", rodada=server_round, loss=loss,
                            accuracy=metricas.get("accuracy"), f1=metricas.get("f1"))
        return loss, metricas


# SEÇÃO 3 - ABRIR A JANELA FLUTUANTE
#=========================================================
def abrir_janela() -> Emissor:
    """
    Inicia a janela em outro processo. A janela escolhe uma porta UDP livre
    e a informa na primeira linha da saída padrão.
    Sem ambiente gráfico (ou sem Tk), devolve um emissor desligado.
    """
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("[visual] Nenhum ambiente gráfico encontrado (DISPLAY vazio). Rodando só no terminal.")
        return Emissor(None)

    fd, controle = tempfile.mkstemp(prefix="fl-visual-", suffix=".json")
    os.close(fd)
    processo = subprocess.Popen(
        [sys.executable, "-m", "visualizador.janela",
         "--codigo", CODIGO_ORIGINAL, "--controle", controle],
        cwd=RAIZ, stdout=subprocess.PIPE, stderr=None, text=True,
        start_new_session=True,   # Ctrl+C no terminal não fecha a janela
    )
    prontos, _, _ = select.select([processo.stdout], [], [], 15)
    linha = processo.stdout.readline().strip() if prontos else ""
    if not linha.startswith("PORTA "):
        print("[visual] Não foi possível abrir a janela (Tkinter instalado? "
              "No Ubuntu: sudo apt install python3-tk). Rodando só no terminal.")
        processo.kill()
        return Emissor(None)
    porta = int(linha.split()[1])
    processo.stdout.close()
    print(f"[visual] Janela de visualização aberta (UDP {porta}).")
    return Emissor(porta, controle)


# SEÇÃO 4 - MAIN (espelho de machine_learning.main)
#=========================================================
def main(amostrar_a_cada: Optional[int] = None, janela: bool = True):
    emissor = abrir_janela() if janela else Emissor(None)

    # Os workers do Ray precisam conseguir importar `visualizador` e
    # `machine_learning`: deixamos a raiz do projeto no PYTHONPATH deles.
    os.environ["PYTHONPATH"] = os.pathsep.join(
        p for p in (RAIZ, os.environ.get("PYTHONPATH")) if p)
    os.chdir(RAIZ)   # hmeq.csv é lido por caminho relativo

    emissor.enviar("etapa", nome="dados")
    X_train, X_test, y_train, y_test = ml.carregar_e_preprocessar_dados()
    input_dim = X_train.shape[1]

    X_partes = np.array_split(X_train, ml.NUM_CLIENTES)
    y_partes = np.array_split(y_train, ml.NUM_CLIENTES)

    print(f"Dados distribuídos entre {ml.NUM_CLIENTES} clientes:")
    for i, (xp, yp) in enumerate(zip(X_partes, y_partes)):
        print(f"  Cliente {i}: {len(xp)} amostras de treino")

    colunas = [c for c in pd.read_csv(ml.CAMINHO_CSV, nrows=0).columns if c != "BAD"]
    arquitetura = [input_dim] + [m.out_features for m in ml.RedeNeural(input_dim).modules()
                                 if isinstance(m, torch.nn.Linear)]
    emissor.enviar(
        "ola", colunas=colunas, arquitetura=arquitetura,
        clientes=ml.NUM_CLIENTES, rodadas=ml.NUM_ROUNDS, epocas=ml.EPOCHS_LOCAL,
        amostras=[len(p) for p in X_partes],
        positivos=[int(p.sum()) for p in y_partes],
        lotes_por_epoca=[int(np.ceil(len(p) / ml.BATCH_SIZE)) for p in X_partes],
        batch=ml.BATCH_SIZE, lr=ml.LEARNING_RATE,
    )

    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)
        ds_treino = TensorDataset(
            torch.tensor(X_partes[idx]).float(),
            torch.tensor(y_partes[idx]).float()
        )
        ds_teste = TensorDataset(
            torch.tensor(X_test).float(),
            torch.tensor(y_test).float()
        )
        loader_treino = DataLoader(ds_treino, batch_size=ml.BATCH_SIZE, shuffle=True)
        loader_teste  = DataLoader(ds_teste,  batch_size=ml.BATCH_SIZE)
        return ClienteVisual(loader_treino, loader_teste, input_dim,
                             idx, emissor, amostrar_a_cada).to_client()

    estrategia = FedAvgVisual(
        fraction_fit=1.0,
        min_fit_clients=ml.NUM_CLIENTES,
        min_available_clients=ml.NUM_CLIENTES,
        evaluate_metrics_aggregation_fn=ml.media_ponderada,
        emissor=emissor,
    )

    print(f"\n{'='*60}")
    print(" Iniciando Aprendizado Federado (com visualização)")
    print(f"   Clientes : {ml.NUM_CLIENTES}")
    print(f"   Rounds   : {ml.NUM_ROUNDS}")
    print(f"   Épocas   : {ml.EPOCHS_LOCAL} por round por cliente")
    print(f"{'='*60}\n")

    emissor.enviar("etapa", nome="simulacao")
    historico = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=ml.NUM_CLIENTES,
        config=fl.server.ServerConfig(num_rounds=ml.NUM_ROUNDS),
        strategy=estrategia,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    print(f"\n{'='*60}")
    print(" RESULTADOS FINAIS DA SIMULAÇÃO")
    print(f"{'='*60}")
    print(historico)
    if historico.metrics_distributed:
        print(f"\n Métricas do último round (round {ml.NUM_ROUNDS}):")
        for nome, lista_valores in historico.metrics_distributed.items():
            print(f"   {nome:12s} = {lista_valores[-1][1]:.4f}")

    emissor.enviar("fim")
    if emissor.ativo:
        print("\n[visual] Treino concluído. A janela continua aberta para você explorar.")
