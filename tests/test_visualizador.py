"""Testes da visualização ao vivo (sem abrir janela gráfica)."""

import importlib.util
import json
import os
import socket
import tempfile
import unittest

from visualizador.protocolo import Controle, Emissor, codificar, decodificar, TAMANHO_MAXIMO

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_FLWR = importlib.util.find_spec("flwr") is not None


class EmissorFalso(Emissor):
    """Guarda os eventos em memória em vez de enviar por UDP."""

    def __init__(self):
        super().__init__(porta=None)
        self.eventos = []

    @property
    def ativo(self):
        return True

    def enviar(self, tipo, **dados):
        self.eventos.append({"tipo": tipo, **dados})


class ProtocoloTests(unittest.TestCase):
    def test_controle_ida_e_volta(self):
        with tempfile.TemporaryDirectory() as d:
            caminho = os.path.join(d, "controle.json")
            janela, treino = Controle(caminho), Controle(caminho)
            janela.escrever(pausado=True, velocidade=4.0)
            self.assertEqual(treino.ler(), {"pausado": True, "velocidade": 4.0})
            self.assertTrue(treino.janela_aberta)
            janela.apagar()
            self.assertFalse(treino.ler()["pausado"])
            self.assertFalse(treino.janela_aberta)

    def test_janela_fechada_nao_trava_o_treino(self):
        controle = Controle(os.path.join(tempfile.gettempdir(), "nao-existe-fl-visual.json"))
        controle.aguardar(duracao_base=60)   # retornaria só após 60 s se esperasse
        self.assertFalse(controle.janela_aberta)

    def test_emissor_desligado_sem_porta(self):
        emissor = Emissor(None)
        self.assertFalse(emissor.ativo)
        emissor.enviar("lote", x=1)   # não deve levantar erro

    def test_udp_local(self):
        receptor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receptor.bind(("127.0.0.1", 0))
        receptor.settimeout(2)
        with tempfile.TemporaryDirectory() as d:
            caminho = os.path.join(d, "c.json")
            Controle(caminho).escrever(False, 1.0)
            emissor = Emissor(receptor.getsockname()[1], caminho)
            emissor.enviar("rodada", rodada=3)
            evento = decodificar(receptor.recv(65535))
        receptor.close()
        self.assertEqual((evento["tipo"], evento["rodada"]), ("rodada", 3))

    def test_lote_realista_cabe_num_datagrama(self):
        pesos = [[[0.123456] * 12] * 16, [[0.123456] * 16] * 8, [[0.123456] * 8]]
        evento = {"tipo": "lote", "pesos": pesos, "gradientes": pesos, "ativacoes": [[0.5] * 16] * 4}
        self.assertLess(len(codificar(evento)), TAMANHO_MAXIMO)
        self.assertEqual(json.loads(codificar(evento))["tipo"], "lote")


@unittest.skipUnless(HAS_TORCH and HAS_FLWR, "requer torch e flwr")
class ObservadorTests(unittest.TestCase):
    def setUp(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        import machine_learning as ml

        self.torch, self.ml = torch, ml
        g = torch.Generator().manual_seed(0)
        self.X = torch.randn(100, 12, generator=g)
        self.y = (torch.rand(100, generator=g) > 0.7).float()
        self.loader = DataLoader(TensorDataset(self.X, self.y), batch_size=32, shuffle=False)

    def _modelo(self):
        self.torch.manual_seed(7)
        return self.ml.RedeNeural(12)

    def test_treino_original_identico_com_e_sem_observador(self):
        from visualizador.observador import ObservadorTreino

        sem = self._modelo()
        self.ml.treinar(sem, self.loader, epocas=2)

        com = self._modelo()
        emissor = EmissorFalso()
        with ObservadorTreino(com, emissor, cliente=1, rodada=2, lotes_por_epoca=len(self.loader),
                              amostrar_a_cada=1):
            self.ml.treinar(com, self.loader, epocas=2)

        for a, b in zip(sem.state_dict().values(), com.state_dict().values()):
            self.assertTrue(self.torch.equal(a, b))
        self.assertEqual(len(emissor.eventos), 2 * len(self.loader))

    def test_fotografia_tem_valores_reais_do_lote(self):
        from visualizador.observador import ObservadorTreino

        modelo = self._modelo()
        emissor = EmissorFalso()
        with ObservadorTreino(modelo, emissor, cliente=0, rodada=1, lotes_por_epoca=len(self.loader),
                              amostrar_a_cada=len(self.loader)):
            self.ml.treinar(modelo, self.loader, epocas=3)

        self.assertEqual([e["epoca"] for e in emissor.eventos], [0, 1, 2])   # 1 lote por época
        primeiro = emissor.eventos[0]
        self.assertEqual(primeiro["lote"], 0)
        self.assertEqual([len(a) for a in primeiro["ativacoes"]], [12, 16, 8, 1])
        for v, esperado in zip(primeiro["ativacoes"][0], self.X[0].tolist()):
            self.assertAlmostEqual(v, esperado, places=3)       # 1ª amostra do 1º lote
        self.assertEqual(primeiro["rotulo"], float(self.y[0]))
        self.assertTrue(all(v >= 0 for v in primeiro["ativacoes"][1]))   # saída de ReLU
        self.assertTrue(0 <= primeiro["ativacoes"][3][0] <= 1)           # saída da Sigmoid
        self.assertGreater(primeiro["loss"], 0)
        self.assertEqual([len(g) for g in primeiro["gradientes"]], [16, 8, 1])

    def test_hooks_globais_removidos_ao_sair(self):
        from visualizador.observador import ObservadorTreino

        modelo = self._modelo()
        emissor = EmissorFalso()
        with ObservadorTreino(modelo, emissor, 0, 1, len(self.loader), 1):
            pass
        self.ml.treinar(modelo, self.loader, epocas=1)
        self.assertEqual(emissor.eventos, [])

    def test_ignora_otimizador_de_outro_modelo(self):
        from visualizador.observador import ObservadorTreino

        observado, outro = self._modelo(), self._modelo()
        emissor = EmissorFalso()
        with ObservadorTreino(observado, emissor, 0, 1, len(self.loader), 1):
            self.ml.treinar(outro, self.loader, epocas=1)
        self.assertEqual(emissor.eventos, [])


if __name__ == "__main__":
    unittest.main()
