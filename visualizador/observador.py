# OBSERVADOR DO TREINO (HOOKS DO PYTORCH)
#=========================================================
# Ideia central: NÃO mexer no código original de machine_learning.py.
# A função `treinar()` original roda exatamente como foi escrita; só
# "escutamos" o que acontece dentro dela usando os hooks do PyTorch:
#
#   forward_pre_hook nas camadas Linear -> valor que ENTRA em cada camada
#                                          (= ativação da camada anterior)
#   forward_hook no modelo              -> saída Sigmoid = P(inadimplente)
#   hook global de módulos (BCELoss)    -> valor da loss e rótulo real
#   hook global pós-step do otimizador  -> momento em que os pesos mudaram
#                                          (gradientes ainda estão em .grad)
#
# A cada N lotes, o observador monta uma "fotografia" do lote (ativações de
# uma amostra, loss, gradientes e pesos novos) e envia para a janela.
#=========================================================

from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim.optimizer import register_optimizer_step_post_hook
from torch.nn.modules.module import register_module_forward_hook

from .protocolo import Emissor, arredondar


class ObservadorTreino:
    """Instala hooks em UM modelo durante o fit() de UM cliente."""

    def __init__(self, modelo: nn.Module, emissor: Emissor, cliente: int,
                 rodada: int, lotes_por_epoca: int, amostrar_a_cada: int):
        self.modelo = modelo
        self.emissor = emissor
        self.cliente = cliente
        self.rodada = rodada
        self.lotes_por_epoca = max(1, lotes_por_epoca)
        self.amostrar_a_cada = max(1, amostrar_a_cada)

        self.camadas: List[nn.Linear] = [m for m in modelo.modules() if isinstance(m, nn.Linear)]
        self.parametros = {id(p) for p in modelo.parameters()}
        self.passos = 0                       # nº de otimizador.step() já feitos
        self._entradas: List[Optional[torch.Tensor]] = [None] * len(self.camadas)
        self._saida: Optional[torch.Tensor] = None
        self._loss: Optional[float] = None
        self._rotulo: Optional[float] = None
        self._handles = []

    # -- ciclo de vida ----------------------------------------------------
    def __enter__(self):
        for i, camada in enumerate(self.camadas):
            self._handles.append(camada.register_forward_pre_hook(self._capturar_entrada(i)))
        self._handles.append(self.modelo.register_forward_hook(self._capturar_saida))
        self._handles.append(register_module_forward_hook(self._capturar_loss))
        self._handles.append(register_optimizer_step_post_hook(self._apos_step))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    # -- hooks ---------------------------------------------------------------
    def _capturar_entrada(self, indice: int):
        def hook(modulo, entradas):
            if self.modelo.training:
                self._entradas[indice] = entradas[0]
        return hook

    def _capturar_saida(self, modulo, entradas, saida):
        if self.modelo.training:
            self._saida = saida

    def _capturar_loss(self, modulo, entradas, saida):
        # Hook global: só nos interessa a BCELoss calculada sobre a saída
        # do NOSSO modelo (o mesmo tensor capturado em _capturar_saida).
        if isinstance(modulo, nn.BCELoss) and self._saida is not None and entradas[0] is self._saida:
            self._loss = float(saida.detach())
            self._rotulo = float(entradas[1].detach().view(-1)[0])

    def _apos_step(self, otimizador, args, kwargs):
        # Hook global: confere se o otimizador é o do NOSSO modelo.
        if not any(id(p) in self.parametros for g in otimizador.param_groups for p in g["params"]):
            return
        self.passos += 1
        indice = self.passos - 1
        lote = indice % self.lotes_por_epoca
        if lote % self.amostrar_a_cada != 0:
            return
        if not self.emissor.ativo:
            return
        self.emissor.enviar("lote", **self.fotografia(indice))
        self.emissor.controle.aguardar()   # deixa a animação acompanhar o código

    # -- fotografia de um lote ----------------------------------------------
    def fotografia(self, indice_passo: int) -> dict:
        with torch.no_grad():
            ativacoes = [e.detach()[0].view(-1).tolist() for e in self._entradas if e is not None]
            ativacoes.append(self._saida.detach()[0].view(-1).tolist())
            pesos = [c.weight.detach().tolist() for c in self.camadas]
            gradientes = [
                c.weight.grad.detach().tolist() if c.weight.grad is not None else None
                for c in self.camadas
            ]
            vieses = [c.bias.detach().tolist() if c.bias is not None else None for c in self.camadas]
        return {
            "cliente": self.cliente,
            "rodada": self.rodada,
            "epoca": indice_passo // self.lotes_por_epoca,
            "lote": indice_passo % self.lotes_por_epoca,
            "lotes_por_epoca": self.lotes_por_epoca,
            "intervalo": self.amostrar_a_cada,
            "loss": self._loss,
            "rotulo": self._rotulo,
            "ativacoes": arredondar(ativacoes),
            "pesos": arredondar(pesos),
            "vieses": arredondar(vieses),
            "gradientes": arredondar(gradientes, 5),
        }
