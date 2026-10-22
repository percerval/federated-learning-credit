# PROTOCOLO ENTRE O TREINO E A JANELA DE VISUALIZAÇÃO
#=========================================================
# O treino (servidor + clientes Flower, que rodam em processos separados do
# Ray) e a janela gráfica são PROCESSOS DIFERENTES. Eles conversam assim:
#
#   treino ──(datagramas UDP com JSON)──▶ janela     (eventos: lote, FedAvg...)
#   treino ◀──(arquivo JSON de controle)── janela     (pausa e velocidade)
#
# Por que UDP?
#   - "Dispara e esquece": enviar um evento nunca trava o treino, mesmo que a
#     janela esteja ocupada desenhando ou tenha sido fechada.
#   - Funciona de qualquer processo (inclusive dos workers do Ray) sem
#     precisar compartilhar filas ou objetos Python.
#
# Por que um arquivo de controle?
#   - Os workers do Ray só precisam ler um arquivo pequeno de vez em quando
#     para saber se devem pausar ou desacelerar. Sem sockets abertos para
#     escuta dentro dos workers.
#=========================================================

import json
import os
import socket
import time
from typing import Any, Dict, Optional

HOST = "127.0.0.1"

# Tempo (em segundos, na velocidade 1x) que a janela leva para animar UM lote
# amostrado: forward camada a camada, loss, backpropagation e atualização dos
# pesos. O treino espera esse mesmo tempo para que a animação acompanhe o
# código "ao vivo" em vez de ficar atrasada.
DURACAO_LOTE = 2.4

# Limite seguro para um datagrama UDP local.
TAMANHO_MAXIMO = 60_000


def arredondar(valor: Any, casas: int = 4) -> Any:
    """Arredonda listas aninhadas de floats para reduzir o tamanho do JSON."""
    if isinstance(valor, float):
        return round(valor, casas)
    if isinstance(valor, (list, tuple)):
        return [arredondar(v, casas) for v in valor]
    return valor


def codificar(evento: Dict[str, Any]) -> bytes:
    dados = json.dumps(evento, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(dados) > TAMANHO_MAXIMO:
        raise ValueError(f"evento '{evento.get('tipo')}' grande demais para UDP ({len(dados)} bytes)")
    return dados


def decodificar(dados: bytes) -> Dict[str, Any]:
    return json.loads(dados.decode("utf-8"))


class Controle:
    """
    Lê (lado do treino) e escreve (lado da janela) o arquivo de controle.

    Conteúdo: {"pausado": bool, "velocidade": float}
    Arquivo ausente = janela fechada -> o treino segue em velocidade máxima.
    """

    def __init__(self, caminho: Optional[str]):
        self.caminho = caminho
        self._mtime = None
        self._estado = {"pausado": False, "velocidade": 1.0}
        self.janela_aberta = caminho is not None

    # -- lado da janela --------------------------------------------------
    def escrever(self, pausado: bool, velocidade: float) -> None:
        if not self.caminho:
            return
        temporario = self.caminho + ".tmp"
        with open(temporario, "w", encoding="utf-8") as f:
            json.dump({"pausado": pausado, "velocidade": velocidade}, f)
        os.replace(temporario, self.caminho)  # troca atômica

    def apagar(self) -> None:
        if self.caminho and os.path.exists(self.caminho):
            os.remove(self.caminho)

    # -- lado do treino --------------------------------------------------
    def ler(self) -> Dict[str, Any]:
        """Relê o arquivo só quando ele muda (custo: um stat por chamada)."""
        if not self.caminho:
            self.janela_aberta = False
            return self._estado
        try:
            mtime = os.stat(self.caminho).st_mtime_ns
        except FileNotFoundError:
            self.janela_aberta = False
            return {"pausado": False, "velocidade": float("inf")}
        if mtime != self._mtime:
            try:
                with open(self.caminho, encoding="utf-8") as f:
                    self._estado = json.load(f)
                self._mtime = mtime
            except (OSError, ValueError):
                pass  # arquivo sendo trocado; usa o estado anterior
        self.janela_aberta = True
        return self._estado

    def aguardar(self, duracao_base: float = DURACAO_LOTE) -> None:
        """
        Segura o treino enquanto a janela estiver em pausa e depois espera o
        tempo da animação (duracao_base / velocidade).
        """
        estado = self.ler()
        while estado.get("pausado") and self.janela_aberta:
            time.sleep(0.1)
            estado = self.ler()
        velocidade = float(estado.get("velocidade", 1.0) or 1.0)
        if self.janela_aberta and velocidade < 50:
            time.sleep(duracao_base / velocidade)


class Emissor:
    """
    Envia eventos para a janela. É "picklable" (o socket é criado sob
    demanda), então pode ser levado para dentro dos workers do Ray.
    """

    def __init__(self, porta: Optional[int], caminho_controle: Optional[str] = None):
        self.porta = porta
        self.controle = Controle(caminho_controle)
        self._sock = None

    def __getstate__(self):
        estado = self.__dict__.copy()
        estado["_sock"] = None
        return estado

    @property
    def ativo(self) -> bool:
        return self.porta is not None and self.controle.janela_aberta

    def enviar(self, tipo: str, **dados: Any) -> None:
        if not self.ativo:
            return
        if self._sock is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        evento = {"tipo": tipo, "t": time.time(), **dados}
        try:
            self._sock.sendto(codificar(evento), (HOST, self.porta))
        except OSError:
            # Janela fechada no meio do treino: o treino continua normalmente.
            pass
