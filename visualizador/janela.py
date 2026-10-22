# JANELA FLUTUANTE DE VISUALIZAÇÃO (Tkinter)
#=========================================================
# Processo separado do treino. Recebe eventos UDP (veja protocolo.py) e
# mostra, sincronizados:
#   - à esquerda: o código ORIGINAL de machine_learning.py, com a linha
#     que acabou de executar destacada;
#   - à direita: a rede neural 12 → 16 → 8 → 1 animada (forward, loss,
#     backpropagation e atualização dos pesos);
#   - embaixo: a federação (servidor ↔ clientes, FedAvg) e as métricas;
#   - rodapé: uma explicação didática do que está acontecendo, com os
#     números reais do lote.
#
# Uso direto (normalmente é aberto por `python3 -m visualizador`):
#   python3 -m visualizador.janela --codigo machine_learning.py --controle /tmp/x.json
#=========================================================

import argparse
import io
import keyword
import math
import queue
import socket
import threading
import time
import tokenize
import tkinter as tk
import tkinter.font as tkfont
from collections import deque

from .protocolo import DURACAO_LOTE, HOST, Controle, decodificar

# ---------------------------------------------------------------- tema ---
COR = {
    "fundo": "#0e131f", "painel": "#151c2b", "borda": "#27314a",
    "texto": "#dbe3f4", "texto2": "#8a96b2", "texto3": "#5b6680",
    "positivo": "#4ea8ff", "negativo": "#ff8a3d", "ativo": "#ffd166",
    "gradiente": "#ff4d8d", "ok": "#3ddc97", "global": "#b388ff",
    "linha_atual": "#3d3514", "bloco": "#172036",
    # sintaxe
    "s_kw": "#c792ea", "s_str": "#c3e88d", "s_com": "#5f6b8a", "s_num": "#f78c6c",
    "s_def": "#82aaff", "s_builtin": "#89ddff",
}

TOKENS_FSTRING = {getattr(tokenize, n) for n in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END")
                  if hasattr(tokenize, n)}

VELOCIDADES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1000.0]   # 1000 = "máx"

# Linhas do código original que cada fase destaca (procuradas pelo texto,
# então continuam corretas mesmo se o arquivo ganhar/perder linhas).
ANCORAS = {
    "dados": "def carregar_e_preprocessar_dados",
    "simulacao": "historico = fl.simulation.start_simulation(",
    "rodada": "config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),",
    "recebe": "self.set_parameters(parametros)",
    "entrada": "for X_batch, y_batch in loader:",
    "camada0": "x = torch.relu(self.fc1(x))",
    "camada1": "x = torch.relu(self.fc2(x))",
    "camada2": "x = self.sigmoid(self.fc3(x))",
    "loss": "loss = criterio(predicoes, y_batch.view(-1, 1))",
    "backward2": "loss.backward()",
    "backward1": "loss.backward()",
    "backward0": "loss.backward()",
    "step": "otimizador.step()",
    "envio": "return self.get_parameters(config={}), len(self.loader_treino.dataset), {}",
    "fedavg": "estrategia = fl.server.strategy.FedAvg(",
    "avaliacao": "acuracia_global = sum(",
    "fim": "RESULTADOS FINAIS DA SIMULAÇÃO",
}

FASES_LOTE = [
    ("entrada", 0.30), ("camada0", 0.30), ("camada1", 0.30), ("camada2", 0.28),
    ("loss", 0.32), ("backward2", 0.22), ("backward1", 0.22), ("backward0", 0.22),
    ("step", 0.24),
]
assert abs(sum(d for _, d in FASES_LOTE) - DURACAO_LOTE) < 1e-6

TITULOS = {
    "dados": "Pré-processamento", "simulacao": "Iniciando a federação",
    "rodada": "Servidor → clientes", "entrada": "Entrada do lote",
    "camada0": "Forward · camada oculta 1", "camada1": "Forward · camada oculta 2",
    "camada2": "Forward · saída", "loss": "Cálculo do erro (loss)",
    "backward2": "Backpropagation", "backward1": "Backpropagation",
    "backward0": "Backpropagation", "step": "Atualização dos pesos",
    "envio": "Cliente → servidor", "fedavg": "Agregação FedAvg",
    "avaliacao": "Avaliação global", "fim": "Treino concluído", "espera": "Aguardando o treino",
}


# ------------------------------------------------------------- utilidades ---
# Sem Xft (ex.: alguns Pythons compilados à parte) o Tk usa fontes X antigas,
# que não têm setas, Σ, subscritos etc. Nesse caso trocamos por ASCII.
SUBSTITUTOS = {"→": "->", "←": "<-", "−": "-", "Σ": "soma", "ₖ": "k", "∂": "d", "⏸": "||",
               "▶": ">", "◀": "<", "✓": "ok", "…": "...", "ᵢ": "i"}
_SEM_UNICODE = False


def seguro(texto: str) -> str:
    if not _SEM_UNICODE:
        return texto
    for k, v in SUBSTITUTOS.items():
        texto = texto.replace(k, v)
    return texto


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def misturar(c1, c2, t):
    """Mistura duas cores hex (t=0 -> c1, t=1 -> c2). Tk não tem transparência."""
    t = max(0.0, min(1.0, t))
    a, b = _rgb(c1), _rgb(c2)
    return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def suave(p):
    """Easing (começa e termina devagar)."""
    p = max(0.0, min(1.0, p))
    return p * p * (3 - 2 * p)


def maximo_abs(matriz):
    m = 0.0
    for linha in matriz or []:
        for v in linha:
            if abs(v) > m:
                m = abs(v)
    return m or 1.0


# =================================================================== app ===
class Janela:
    def __init__(self, raiz: tk.Tk, codigo: str, controle: Controle):
        self.raiz = raiz
        self.controle = controle
        self.eventos: "queue.Queue[dict]" = queue.Queue()

        # Estado do treino
        self.cfg = {"arquitetura": [12, 16, 8, 1], "colunas": [f"x{i}" for i in range(12)],
                    "clientes": 3, "rodadas": 10, "epocas": 5, "amostras": [0, 0, 0],
                    "lotes_por_epoca": [1, 1, 1], "batch": 32, "lr": 0.01}
        self.foco = 0                       # cliente mostrado na rede grande
        self.seguir_auto = True             # troca o foco para quem estiver treinando
        self.rodada = 0
        self.metricas = []                  # [(rodada, acc, f1, loss)]
        self.cartoes = {}                   # cliente -> dict(estado, epoca, lote, loss)
        self.ultimo_lote = {}               # cliente -> evento lote
        self.pesos_vistos = None            # pesos atualmente desenhados
        self.titulo_rede = "Rede neural"
        self.concluido = False

        # Linha do tempo de animação
        self.fila = deque()
        self.atual = None                   # item em animação
        self.fase_atual = None
        self.pausado = False
        self.vel_idx = VELOCIDADES.index(1.0)
        self.ultimo_tick = time.time()
        self.voos = []                      # pontos voando na federação

        self._fontes()
        self._montar_layout()
        self._carregar_codigo(codigo)
        self._salvar_controle()
        self._explicar("espera", "Abra o treino com  python3 -m visualizador  — os eventos aparecem aqui.")
        self.raiz.after(33, self._tick)

    # ----------------------------------------------------------- layout ---
    def _fontes(self):
        familias = set(tkfont.families(self.raiz))
        mono = next((f for f in ("JetBrains Mono", "Fira Code", "DejaVu Sans Mono", "Ubuntu Mono",
                                 "Liberation Mono") if f in familias), "TkFixedFont")
        sans = next((f for f in ("Inter", "Ubuntu", "DejaVu Sans", "Liberation Sans") if f in familias),
                    "TkDefaultFont")
        self.tam_codigo = 10
        self.f_codigo = tkfont.Font(family=mono, size=self.tam_codigo)
        self.f_mono_p = tkfont.Font(family=mono, size=8)
        self.f_ui = tkfont.Font(family=sans, size=10)
        self.f_ui_b = tkfont.Font(family=sans, size=10, weight="bold")
        self.f_ui_p = tkfont.Font(family=sans, size=8)
        self.f_titulo = tkfont.Font(family=sans, size=13, weight="bold")
        self.f_fase = tkfont.Font(family=sans, size=12, weight="bold")

    def _botao(self, pai, texto, comando, largura=None):
        b = tk.Button(pai, text=seguro(texto), command=comando, bg=COR["borda"], fg=COR["texto"],
                      activebackground=COR["texto3"], activeforeground=COR["texto"],
                      relief="flat", bd=0, padx=10, pady=4, font=self.f_ui, highlightthickness=0,
                      cursor="hand2")
        if largura:
            b.configure(width=largura)
        return b

    def _montar_layout(self):
        r = self.raiz
        r.title("Aprendizado Federado · visualização ao vivo")
        r.configure(bg=COR["fundo"])
        r.geometry("1440x880")
        r.minsize(1100, 700)
        r.columnconfigure(0, weight=5, uniform="col")
        r.columnconfigure(1, weight=7, uniform="col")
        r.rowconfigure(1, weight=1)

        # -- cabeçalho
        topo = tk.Frame(r, bg=COR["fundo"])
        topo.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 6))
        tk.Label(topo, text="Aprendizado Federado · HMEQ", font=self.f_titulo,
                 bg=COR["fundo"], fg=COR["texto"]).pack(side="left")
        self.lbl_status = tk.Label(topo, text="aguardando o treino…", font=self.f_ui,
                                   bg=COR["fundo"], fg=COR["texto2"])
        self.lbl_status.pack(side="left", padx=18)

        self.var_topo = tk.BooleanVar(value=True)
        tk.Checkbutton(topo, text="Sempre no topo", variable=self.var_topo, command=self._topo,
                       bg=COR["fundo"], fg=COR["texto2"], selectcolor=COR["painel"],
                       activebackground=COR["fundo"], activeforeground=COR["texto"],
                       font=self.f_ui, highlightthickness=0, bd=0).pack(side="right", padx=(10, 0))
        self._botao(topo, "A+", lambda: self._zoom(+1)).pack(side="right", padx=2)
        self._botao(topo, "A−", lambda: self._zoom(-1)).pack(side="right", padx=2)
        tk.Frame(topo, width=12, bg=COR["fundo"]).pack(side="right")
        self._botao(topo, "▶▶", lambda: self._velocidade(+1)).pack(side="right", padx=2)
        self.lbl_vel = tk.Label(topo, text="1x", width=5, font=self.f_ui_b,
                                bg=COR["fundo"], fg=COR["ativo"])
        self.lbl_vel.pack(side="right")
        self._botao(topo, "◀◀", lambda: self._velocidade(-1)).pack(side="right", padx=2)
        tk.Label(topo, text="velocidade", font=self.f_ui_p, bg=COR["fundo"],
                 fg=COR["texto3"]).pack(side="right", padx=(12, 4))
        self.btn_pausa = self._botao(topo, "⏸  Pausar", self._alternar_pausa, largura=11)
        self.btn_pausa.pack(side="right", padx=4)
        self._topo()

        # -- código (esquerda)
        esq = tk.Frame(r, bg=COR["painel"], highlightbackground=COR["borda"], highlightthickness=1)
        esq.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=(12, 6), pady=6)
        esq.rowconfigure(1, weight=1)
        esq.columnconfigure(0, weight=1)
        tk.Label(esq, text="machine_learning.py  ·  código original em execução", anchor="w",
                 font=self.f_ui_b, bg=COR["painel"], fg=COR["texto2"]).grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(8, 4))
        self.txt = tk.Text(esq, bg="#0f1524", fg=COR["texto"], insertbackground=COR["texto"],
                           font=self.f_codigo, wrap="none", bd=0, padx=6, pady=6,
                           highlightthickness=0, cursor="arrow")
        barra = tk.Scrollbar(esq, command=self.txt.yview, bg=COR["painel"],
                             troughcolor=COR["painel"], bd=0, highlightthickness=0)
        self.txt.configure(yscrollcommand=barra.set)
        self.txt.grid(row=1, column=0, sticky="nsew", padx=(6, 0), pady=(0, 6))
        barra.grid(row=1, column=1, sticky="ns", pady=(0, 6))

        # -- rede neural (direita, em cima)
        dir_ = tk.Frame(r, bg=COR["fundo"])
        dir_.grid(row=1, column=1, sticky="nsew", padx=(6, 12), pady=6)
        dir_.columnconfigure(0, weight=1)
        dir_.columnconfigure(1, weight=1)
        dir_.rowconfigure(0, weight=3)
        dir_.rowconfigure(1, weight=2)
        self.cv_rede = tk.Canvas(dir_, bg=COR["painel"], highlightbackground=COR["borda"],
                                 highlightthickness=1, bd=0)
        self.cv_rede.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 6))
        self.cv_fed = tk.Canvas(dir_, bg=COR["painel"], highlightbackground=COR["borda"],
                                highlightthickness=1, bd=0)
        self.cv_fed.grid(row=1, column=0, sticky="nsew", padx=(0, 3))
        self.cv_met = tk.Canvas(dir_, bg=COR["painel"], highlightbackground=COR["borda"],
                                highlightthickness=1, bd=0)
        self.cv_met.grid(row=1, column=1, sticky="nsew", padx=(3, 0))
        self.cv_rede.bind("<Configure>", lambda e: self._construir_rede())
        self.cv_fed.bind("<Configure>", lambda e: self._construir_federacao())
        self.cv_met.bind("<Configure>", lambda e: self._desenhar_metricas())

        # -- explicação (rodapé)
        rod = tk.Frame(r, bg=COR["painel"], highlightbackground=COR["borda"], highlightthickness=1)
        rod.grid(row=2, column=1, sticky="ew", padx=(6, 12), pady=(0, 12))
        self.lbl_fase = tk.Label(rod, text="", font=self.f_fase, bg=COR["painel"],
                                 fg=COR["ativo"], anchor="w")
        self.lbl_fase.pack(fill="x", padx=12, pady=(8, 0))
        self.lbl_expl = tk.Label(rod, text="", font=self.f_ui, bg=COR["painel"], fg=COR["texto"],
                                 anchor="w", justify="left", wraplength=760)
        self.lbl_expl.pack(fill="x", padx=12, pady=(2, 10))
        rod.bind("<Configure>", lambda e: self.lbl_expl.configure(wraplength=max(300, e.width - 30)))
        r.rowconfigure(2, minsize=96)

    # ------------------------------------------------------- controles ---
    def _topo(self):
        self.raiz.attributes("-topmost", bool(self.var_topo.get()))

    def _zoom(self, passo):
        self.tam_codigo = max(7, min(22, self.tam_codigo + passo))
        self.f_codigo.configure(size=self.tam_codigo)

    def _velocidade(self, passo):
        self.vel_idx = max(0, min(len(VELOCIDADES) - 1, self.vel_idx + passo))
        v = VELOCIDADES[self.vel_idx]
        self.lbl_vel.configure(text="máx" if v >= 1000 else (f"{v:g}x"))
        self._salvar_controle()

    def _alternar_pausa(self):
        self.pausado = not self.pausado
        self.btn_pausa.configure(text=seguro("▶  Continuar" if self.pausado else "⏸  Pausar"),
                                 bg=COR["ativo"] if self.pausado else COR["borda"],
                                 fg=COR["fundo"] if self.pausado else COR["texto"])
        self._salvar_controle()

    def _salvar_controle(self):
        try:
            self.controle.escrever(self.pausado, VELOCIDADES[self.vel_idx])
        except OSError:
            pass

    @property
    def velocidade(self):
        return VELOCIDADES[self.vel_idx]

    # ----------------------------------------------------------- código ---
    def _carregar_codigo(self, caminho):
        try:
            with open(caminho, encoding="utf-8") as f:
                fonte = f.read()
        except OSError:
            fonte = f"# não foi possível abrir {caminho}\n"
        self.linhas = fonte.splitlines()
        t = self.txt
        t.configure(state="normal")
        t.delete("1.0", "end")
        largura = len(str(len(self.linhas)))
        for i, linha in enumerate(self.linhas, 1):
            t.insert("end", f"{i:>{largura}}  ", ("numero",))
            t.insert("end", linha + "\n")
        self.desloc = largura + 2   # colunas ocupadas pela numeração

        for nome, cor in (("s_kw", "s_kw"), ("s_str", "s_str"), ("s_com", "s_com"),
                          ("s_num", "s_num"), ("s_def", "s_def"), ("s_builtin", "s_builtin")):
            t.tag_configure(nome, foreground=COR[cor])
        t.tag_configure("numero", foreground=COR["texto3"])
        t.tag_configure("bloco", background=COR["bloco"])
        t.tag_configure("atual", background=COR["linha_atual"])
        t.tag_raise("atual")
        self._colorir(fonte)
        t.configure(state="disabled")

        # índice: âncora -> número da linha (1ª ocorrência)
        self.linha_de = {}
        for fase, texto in ANCORAS.items():
            for i, linha in enumerate(self.linhas, 1):
                if texto in linha:
                    self.linha_de[fase] = i
                    break

    def _colorir(self, fonte):
        builtins_ = {"print", "len", "range", "sum", "zip", "enumerate", "int", "float", "str",
                     "dict", "list", "tuple", "super", "isinstance", "round"}
        anterior = None
        try:
            for tok in tokenize.generate_tokens(io.StringIO(fonte).readline):
                (l1, c1), (l2, c2) = tok.start, tok.end
                ini = f"{l1}.{c1 + self.desloc}"
                fim = f"{l2}.{c2 + self.desloc}"
                tag = None
                if tok.type == tokenize.COMMENT:
                    tag = "s_com"
                elif tok.type == tokenize.STRING or tok.type in TOKENS_FSTRING:
                    tag = "s_str"
                elif tok.type == tokenize.NUMBER:
                    tag = "s_num"
                elif tok.type == tokenize.NAME:
                    if keyword.iskeyword(tok.string):
                        tag = "s_kw"
                    elif anterior in ("def", "class"):
                        tag = "s_def"
                    elif tok.string in builtins_:
                        tag = "s_builtin"
                if tag:
                    if l1 != l2:   # string em várias linhas: colore linha a linha
                        for ln in range(l1, l2 + 1):
                            a = c1 + self.desloc if ln == l1 else self.desloc
                            b = f"{ln}.{c2 + self.desloc}" if ln == l2 else f"{ln}.end"
                            self.txt.tag_add(tag, f"{ln}.{a}", b)
                    else:
                        self.txt.tag_add(tag, ini, fim)
                if tok.type == tokenize.NAME:
                    anterior = tok.string
                elif tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                                      tokenize.DEDENT, tokenize.COMMENT):
                    anterior = None
        except (tokenize.TokenError, IndentationError):
            pass

    def _bloco_da_linha(self, n):
        """Encontra a função (def) que contém a linha n -> (início, fim)."""
        def indent(s):
            return len(s) - len(s.lstrip())
        ini = n
        alvo = indent(self.linhas[n - 1])
        for i in range(n, 0, -1):
            s = self.linhas[i - 1]
            if s.strip().startswith(("def ", "class ")) and indent(s) < alvo:
                ini = i
                break
        base = indent(self.linhas[ini - 1])
        fim = ini
        for i in range(ini + 1, len(self.linhas) + 1):
            s = self.linhas[i - 1]
            if s.strip() and indent(s) <= base:
                break
            fim = i
        while fim > ini and not self.linhas[fim - 1].strip():
            fim -= 1
        return ini, fim

    def _destacar_codigo(self, fase):
        n = self.linha_de.get(fase)
        if not n:
            return
        t = self.txt
        t.tag_remove("atual", "1.0", "end")
        t.tag_remove("bloco", "1.0", "end")
        a, b = self._bloco_da_linha(n)
        t.tag_add("bloco", f"{a}.0", f"{b + 1}.0")
        t.tag_add("atual", f"{n}.0", f"{n + 1}.0")
        # rola para manter a linha atual perto do meio
        t.see(f"{min(len(self.linhas), n + 8)}.0")
        t.see(f"{max(1, n - 8)}.0")

    # ------------------------------------------------- rede neural (canvas) ---
    def _construir_rede(self):
        cv = self.cv_rede
        cv.delete("all")
        w, h = max(cv.winfo_width(), 400), max(cv.winfo_height(), 250)
        arq = self.cfg["arquitetura"]
        margem_esq, margem_dir, topo, base = 150, 190, 58, 40
        xs = [margem_esq + (w - margem_esq - margem_dir) * i / (len(arq) - 1) for i in range(len(arq))]
        self.pos = []
        maior = max(arq)
        passo = (h - topo - base) / maior
        self.raio = max(4, min(13, passo * 0.38))
        for l, n in enumerate(arq):
            altura = passo * (n - 1)
            y0 = topo + (h - topo - base - altura) / 2
            self.pos.append([(xs[l], y0 + passo * i) for i in range(n)])

        # arestas (desenhadas primeiro, ficam atrás dos neurônios)
        self.it_arestas = []
        for l in range(len(arq) - 1):
            camada = []
            for j, (x1, y1) in enumerate(self.pos[l + 1]):
                camada.append([cv.create_line(x0, y0, x1, y1, fill=COR["borda"], width=1)
                               for (x0, y0) in self.pos[l]])
            self.it_arestas.append(camada)

        self.it_nos = []
        for l, camada in enumerate(self.pos):
            itens = []
            for (x, y) in camada:
                rr = self.raio * (1.5 if l == len(arq) - 1 else 1)
                itens.append(cv.create_oval(x - rr, y - rr, x + rr, y + rr, fill=COR["fundo"],
                                            outline=COR["texto3"], width=1.5))
            self.it_nos.append(itens)
        # pontinhos do sinal/gradiente: criados por último para ficarem por cima
        self.it_pontos = [cv.create_oval(0, 0, 0, 0, fill=COR["ativo"], outline="", state="hidden")
                          for _ in range(14)]

        # rótulos
        nomes = ["Entrada", "Oculta 1 · ReLU", "Oculta 2 · ReLU", "Saída · Sigmoid"]
        for l, n in enumerate(arq):
            nome = nomes[l] if len(arq) == 4 else f"Camada {l}"
            cv.create_text(xs[l], topo - 22, text=f"{nome} ({n})", fill=COR["texto2"],
                           font=self.f_ui_b)
        self.it_rot_entrada = []
        for i, (x, y) in enumerate(self.pos[0]):
            nome = self.cfg["colunas"][i] if i < len(self.cfg["colunas"]) else f"x{i}"
            self.it_rot_entrada.append(cv.create_text(x - self.raio - 8, y, text=nome, anchor="e",
                                                      fill=COR["texto2"], font=self.f_mono_p))
        xo, yo = self.pos[-1][0]
        self.it_saida = cv.create_text(xo + self.raio * 1.5 + 14, yo, anchor="w", text="",
                                       fill=COR["texto"], font=self.f_ui, justify="left")
        self.it_titulo_rede = cv.create_text(14, 14, anchor="nw", text=self.titulo_rede,
                                             fill=COR["texto"], font=self.f_ui_b)
        # legenda
        lx, ly = 14, h - 16
        for cor, texto in ((COR["positivo"], "peso +"), (COR["negativo"], "peso −"),
                           (COR["ativo"], "sinal (forward)"), (COR["gradiente"], "gradiente (backward)")):
            texto = seguro(texto)
            cv.create_line(lx, ly, lx + 18, ly, fill=cor, width=3)
            cv.create_text(lx + 24, ly, anchor="w", text=texto, fill=COR["texto2"], font=self.f_ui_p)
            lx += 30 + self.f_ui_p.measure(texto) + 14
        if self.pesos_vistos:
            self._pintar_pesos(self.pesos_vistos)

    def _cor_peso(self, v, escala, forca=1.0):
        t = min(1.0, abs(v) / escala)
        base = COR["positivo"] if v >= 0 else COR["negativo"]
        return misturar(COR["painel"], base, (0.12 + 0.88 * t) * forca), 0.5 + 2.6 * t * forca

    def _pintar_pesos(self, pesos, forca=0.75):
        cv = self.cv_rede
        for l, W in enumerate(pesos[:len(self.it_arestas)]):
            esc = maximo_abs(W)
            for j, linha in enumerate(W):
                for i, v in enumerate(linha):
                    cor, larg = self._cor_peso(v, esc, forca)
                    cv.itemconfigure(self.it_arestas[l][j][i], fill=cor, width=larg)

    def _pintar_neuronios(self, camada, valores, fracao=1.0, escala=None):
        cv = self.cv_rede
        ult = len(self.it_nos) - 1
        esc = escala or max([abs(v) for v in valores] + [1e-6])
        for i, v in enumerate(valores):
            if camada == ult:   # probabilidade: verde (0) -> rosa (1)
                cor = misturar(COR["ok"], COR["gradiente"], v)
                cor = misturar(COR["fundo"], cor, fracao)
            elif camada == 0:   # entrada normalizada: pode ser negativa
                base = COR["positivo"] if v >= 0 else COR["negativo"]
                cor = misturar(COR["fundo"], base, min(1, abs(v) / 2.5) * fracao)
            else:               # ReLU: 0 = apagado
                cor = misturar(COR["fundo"], COR["ativo"], min(1, abs(v) / esc) * fracao)
            cv.itemconfigure(self.it_nos[camada][i], fill=cor,
                             outline=COR["ativo"] if (v > 0 and fracao > 0.5 and 0 < camada < ult)
                             else COR["texto3"])

    def _limpar_neuronios(self):
        for l, itens in enumerate(self.it_nos):
            for it in itens:
                self.cv_rede.itemconfigure(it, fill=COR["fundo"], outline=COR["texto3"])

    def _pontos(self, arestas, progresso, cor, reverso=False):
        """Pontinhos correndo pelas arestas mais fortes."""
        cv = self.cv_rede
        for k, it in enumerate(self.it_pontos):
            if k >= len(arestas):
                cv.itemconfigure(it, state="hidden")
                continue
            (x0, y0), (x1, y1) = arestas[k]
            p = 1 - progresso if reverso else progresso
            x, y = x0 + (x1 - x0) * p, y0 + (y1 - y0) * p
            r = 3.2
            cv.coords(it, x - r, y - r, x + r, y + r)
            cv.itemconfigure(it, fill=cor, state="normal" if 0.02 < progresso < 0.98 else "hidden")

    def _esconder_pontos(self):
        for it in self.it_pontos:
            self.cv_rede.itemconfigure(it, state="hidden")

    def _titulo_rede(self, texto):
        self.titulo_rede = texto
        if hasattr(self, "it_titulo_rede"):
            self.cv_rede.itemconfigure(self.it_titulo_rede, text=texto)

    # ---------------------------------------------- federação (canvas) ---
    def _construir_federacao(self):
        cv = self.cv_fed
        cv.delete("all")
        w, h = max(cv.winfo_width(), 200), max(cv.winfo_height(), 150)
        n = self.cfg["clientes"]
        cv.create_text(12, 12, anchor="nw", text="Federação", fill=COR["texto"], font=self.f_ui_b)
        cv.create_text(w - 10, 14, anchor="ne", text="clique num cliente para fixá-lo",
                       fill=COR["texto3"], font=self.f_ui_p)
        self.pos_servidor = (w / 2, 62)
        self.pos_clientes = [((i + 0.5) * w / n, h - 82) for i in range(n)]
        sx, sy = self.pos_servidor
        self.it_ligacoes = [cv.create_line(sx, sy, x, y, fill=COR["borda"], width=2, dash=(4, 4))
                            for x, y in self.pos_clientes]
        r = 22
        self.it_servidor = cv.create_rectangle(sx - 46, sy - 18, sx + 46, sy + 18, fill=COR["fundo"],
                                               outline=COR["global"], width=2)
        cv.create_text(sx, sy, text="Servidor", fill=COR["texto"], font=self.f_ui_b)
        self.it_formula = cv.create_text(sx + 56, sy, anchor="w", text=seguro("FedAvg\nw = Σ (nₖ/n)·wₖ"),
                                         fill=COR["texto3"], font=self.f_ui_p, justify="left")
        self.it_cli = []
        for i, (x, y) in enumerate(self.pos_clientes):
            arco = cv.create_arc(x - r - 5, y - r - 5, x + r + 5, y + r + 5, start=90, extent=0,
                                 style="arc", outline=COR["ok"], width=3)
            no = cv.create_oval(x - r, y - r, x + r, y + r, fill=COR["fundo"],
                                outline=COR["texto3"], width=2, tags=(f"cli{i}",))
            txt = cv.create_text(x, y, text=f"C{i}", fill=COR["texto"], font=self.f_ui_b,
                                 tags=(f"cli{i}",))
            amostras = self.cfg["amostras"][i] if i < len(self.cfg["amostras"]) else 0
            info = cv.create_text(x, y + r + 8, anchor="n", text=f"{amostras} amostras", fill=COR["texto2"],
                                  font=self.f_ui_p, justify="center")
            cv.tag_bind(f"cli{i}", "<Button-1>", lambda e, k=i: self._focar(k, manual=True))
            self.it_cli.append({"arco": arco, "no": no, "txt": txt, "info": info})
        self.it_voos = []
        self._atualizar_cartoes()

    def _atualizar_cartoes(self):
        if not hasattr(self, "it_cli"):
            return
        cv = self.cv_fed
        E = self.cfg["epocas"]
        for i, it in enumerate(self.it_cli):
            c = self.cartoes.get(i, {})
            estado = c.get("estado", "aguardando")
            amostras = self.cfg["amostras"][i] if i < len(self.cfg["amostras"]) else 0
            if estado == "treinando" and "epoca" in c:
                lpe = max(1, c.get("lotes_por_epoca", 1))
                prog = (c["epoca"] + (c["lote"] + 1) / lpe) / E
                linha2 = f"época {c['epoca'] + 1}/{E}" + (f"\nloss {c['loss']:.3f}" if c.get("loss") is not None else "")
            elif estado == "enviou":
                prog, linha2 = 1.0, "pesos enviados ✓"
            elif estado == "treinando":
                prog, linha2 = 0.0, "treinando…"
            else:
                prog, linha2 = 0.0, "aguardando"
            cv.itemconfigure(it["arco"], extent=-359.9 * min(1, prog))
            cv.itemconfigure(it["info"], text=seguro(f"{amostras} amostras\n{linha2}"))
            foco = i == self.foco
            cv.itemconfigure(it["no"], outline=COR["ativo"] if foco else COR["texto3"],
                             width=3 if foco else 2, dash=() if (foco and not self.seguir_auto) or not foco
                             else (3, 2))

    def _voar(self, origem, destino, cor, duracao=0.7, atraso=0.0):
        self.voos.append({"o": origem, "d": destino, "cor": cor, "t0": time.time() + atraso,
                          "dur": duracao, "it": None})

    def _animar_voos(self):
        cv = self.cv_fed
        agora = time.time()
        vivos = []
        for v in self.voos:
            p = (agora - v["t0"]) / v["dur"]
            if p >= 1:
                if v["it"]:
                    cv.delete(v["it"])
                continue
            if p >= 0:
                (x0, y0), (x1, y1) = v["o"], v["d"]
                q = suave(p)
                x, y = x0 + (x1 - x0) * q, y0 + (y1 - y0) * q
                r = 6
                if v["it"] is None:
                    v["it"] = cv.create_oval(x - r, y - r, x + r, y + r, fill=v["cor"], outline="")
                else:
                    cv.coords(v["it"], x - r, y - r, x + r, y + r)
            vivos.append(v)
        self.voos = vivos

    def _focar(self, k, manual=False):
        if manual:
            # clicar no cliente já fixado volta para o modo automático
            self.seguir_auto = (k == self.foco and not self.seguir_auto)
            self._atualizar_cartoes()
        if k == self.foco:
            return
        self.foco = k
        # descarta lotes pendentes do cliente anterior
        self.fila = deque(e for e in self.fila if e["tipo"] != "lote")
        self._atualizar_cartoes()
        ev = self.ultimo_lote.get(k)
        if ev:
            self.pesos_vistos = ev["pesos"]
            self._pintar_pesos(ev["pesos"])
        self._titulo_rede(f"Rede local do cliente {k}" + (f" · round {self.rodada}" if self.rodada else ""))

    # --------------------------------------------------- métricas (canvas) ---
    def _desenhar_metricas(self):
        cv = self.cv_met
        cv.delete("all")
        w, h = max(cv.winfo_width(), 200), max(cv.winfo_height(), 150)
        cv.create_text(12, 12, anchor="nw", text="Avaliação global por round", fill=COR["texto"],
                       font=self.f_ui_b)
        x0, x1, y0, y1 = 42, w - 16, 44, h - 30
        R = max(1, self.cfg["rodadas"])
        vmin = 0.0
        for k in range(0, 6):
            v = vmin + (1 - vmin) * k / 5
            y = y1 - (y1 - y0) * (v - vmin) / (1 - vmin)
            cv.create_line(x0, y, x1, y, fill=COR["borda"])
            cv.create_text(x0 - 6, y, anchor="e", text=f"{v:.1f}", fill=COR["texto3"], font=self.f_ui_p)
        for r in range(1, R + 1):
            x = x0 + (x1 - x0) * (r - 1) / max(1, R - 1)
            cv.create_text(x, y1 + 12, text=str(r), fill=COR["texto3"], font=self.f_ui_p)

        def px(r, v):
            return (x0 + (x1 - x0) * (r - 1) / max(1, R - 1),
                    y1 - (y1 - y0) * (min(1, max(vmin, v)) - vmin) / (1 - vmin))

        series = (("acurácia", 1, COR["positivo"], None), ("F1", 2, COR["ativo"], None),
                  ("loss", 3, COR["gradiente"], (4, 3)))
        lx = x1
        for nome, idx, cor, traco in reversed(series):
            pts = [px(m[0], m[idx]) for m in self.metricas if m[idx] is not None]
            if len(pts) >= 2:
                cv.create_line(*[c for p in pts for c in p], fill=cor, width=2, dash=traco)
            for (x, y) in pts:
                cv.create_oval(x - 3, y - 3, x + 3, y + 3, fill=cor, outline="")
            if self.metricas and self.metricas[-1][idx] is not None:
                x, y = pts[-1]
                perto_fim = x > x1 - 50
                cv.create_text(x - 6 if perto_fim else x + 6, y - 9, anchor="e" if perto_fim else "w",
                               text=f"{self.metricas[-1][idx]:.3f}", fill=cor, font=self.f_ui_p)
            texto = nome
            lx -= self.f_ui_p.measure(texto) + 26
            cv.create_line(lx, 20, lx + 14, 20, fill=cor, width=2, dash=traco)
            cv.create_text(lx + 18, 20, anchor="w", text=texto, fill=COR["texto2"], font=self.f_ui_p)
        if not self.metricas:
            cv.create_text((x0 + x1) / 2, (y0 + y1) / 2, text="as métricas aparecem ao fim de cada round",
                           fill=COR["texto3"], font=self.f_ui_p)

    # ---------------------------------------------------- explicações ---
    def _explicar(self, fase, texto):
        self.lbl_fase.configure(text=seguro(TITULOS.get(fase, fase)))
        self.lbl_expl.configure(text=seguro(texto))

    def _texto_fase(self, fase, item):
        c = self.cfg
        if item["tipo"] == "lote":
            ev = item["ev"]
            a = ev["ativacoes"]
            p = a[-1][0]
            y = ev.get("rotulo")
            n_lote, lpe = ev["lote"] + 1, ev["lotes_por_epoca"]
            onde = (f"Cliente {ev['cliente']} · round {ev['rodada']} · época {ev['epoca'] + 1}/{c['epocas']}"
                    f" · lote {n_lote}/{lpe}")
            if fase == "entrada":
                return (f"{onde}. O DataLoader entrega {c['batch']} pessoas do conjunto privado deste "
                        f"cliente; seguimos a 1ª delas. Seus {len(a[0])} atributos (já normalizados: "
                        f"0 = média) entram na rede. Azul = acima da média, laranja = abaixo.")
            if fase.startswith("camada"):
                l = int(fase[-1])
                if l == len(a) - 2:
                    return (f"A saída faz z = Σ wᵢ·hᵢ + b e aplica a Sigmoid, que espreme z para "
                            f"[0, 1]: P(inadimplente) = {p:.3f}. Acima de 0.5 o modelo diria BAD = 1.")
                ativos = sum(1 for v in a[l + 1] if v > 0)
                return (f"Cada um dos {len(a[l + 1])} neurônios calcula z = Σ wᵢ·xᵢ + b e aplica ReLU "
                        f"(max(0, z)). {ativos} de {len(a[l + 1])} ficaram ativos (amarelos). As linhas "
                        f"acesas são as entradas que mais pesaram: |wᵢ·xᵢ| grande.")
            if fase == "loss":
                real = "inadimplente (1)" if y == 1 else "adimplente (0)"
                loss = ev.get("loss")
                return (f"Rótulo real: {real}. Previsto: P = {p:.3f}. A BCE pune a distância entre os dois: "
                        f"−[y·log(P) + (1−y)·log(1−P)]. Média do lote inteiro: loss = {loss:.4f}."
                        if loss is not None else f"Rótulo real: {real}. Previsto: P = {p:.3f}.")
            if fase.startswith("backward"):
                l = int(fase[-1])
                g = ev["gradientes"][l]
                gmax = maximo_abs(g) if g else 0
                return (f"loss.backward() aplica a regra da cadeia da saída para a entrada e calcula "
                        f"∂L/∂w de cada peso. Camada {l + 1}: maior |gradiente| = {gmax:.5f}. "
                        f"Linhas rosa = pesos que mais \"culpam\" o erro.")
            if fase == "step":
                return (f"otimizador.step(): o Adam (lr = {c['lr']}) move cada peso contra o gradiente. "
                        f"Como animamos 1 lote a cada {item.get('intervalo', '?')}, os pesos mostrados "
                        f"agora já incluem todas as atualizações feitas desde o último lote animado.")
        ev = item.get("ev", {})
        if fase == "rodada":
            return (f"Round {ev.get('rodada')}/{c['rodadas']}: o servidor envia o MESMO modelo global aos "
                    f"{c['clientes']} clientes. Cada um treina {c['epocas']} épocas com os próprios dados, "
                    f"que nunca saem do cliente.")
        if fase == "envio":
            return (f"Cliente {ev.get('cliente')} terminou e envia só os PESOS ao servidor "
                    f"(treinou com {ev.get('amostras')} amostras). Nenhum dado bruto é compartilhado.")
        if fase == "fedavg":
            ns = ev.get("amostras") or []
            n = sum(ns) or 1
            partes = " + ".join(f"{k / n:.3f}·w{i}" for i, k in enumerate(ns))
            return (f"FedAvg: média dos pesos ponderada pelo nº de amostras (n = {n}): "
                    f"w_global = {partes}. A rede acima agora mostra o modelo global.")
        if fase == "avaliacao":
            acc, f1, loss = ev.get("accuracy"), ev.get("f1"), ev.get("loss")
            return (f"Round {ev.get('rodada')}: cada cliente avalia o modelo global no conjunto de teste e "
                    f"o servidor faz a média ponderada → acurácia {acc:.1%}, F1 {f1:.3f}, loss {loss:.4f}."
                    if acc is not None else "Avaliação concluída.")
        if fase == "dados":
            return ("Lendo hmeq.csv e preenchendo valores ausentes coluna a coluna em paralelo "
                    "(ProcessPoolExecutor), depois split treino/teste e normalização.")
        if fase == "simulacao":
            return ("O Flower inicia a simulação: o Ray cria processos para os clientes virtuais. "
                    "O treino de cada cliente é a função treinar() original, observada por hooks do PyTorch.")
        if fase == "fim":
            if self.metricas:
                _, acc, f1, _ = self.metricas[-1]
                return (f"Fim dos {c['rodadas']} rounds. Modelo global final: acurácia {acc:.1%}, F1 {f1:.3f}. "
                        f"Os resultados completos estão no terminal. A janela pode ser fechada.")
            return "Fim do treino."
        return ""

    # ------------------------------------------------------- eventos ---
    def receber(self, ev):
        self.eventos.put(ev)

    def _processar(self, ev):
        tipo = ev.get("tipo")
        if tipo == "ola":
            self.cfg.update({k: ev[k] for k in ev if k in self.cfg or k in ("positivos",)})
            self._construir_rede()
            self._construir_federacao()
            self._desenhar_metricas()
            self._titulo_rede(f"Rede local do cliente {self.foco}")
        elif tipo == "etapa":
            self.fila.append({"tipo": "etapa", "fases": [(ev["nome"], 0.8)], "ev": ev})
        elif tipo == "rodada":
            self.rodada = ev["rodada"]
            for i in range(self.cfg["clientes"]):
                self.cartoes[i] = {"estado": "aguardando"}
            self.fila.append({"tipo": "rodada", "fases": [("rodada", 0.9)], "ev": ev})
        elif tipo == "recebeu":
            self.cartoes[ev["cliente"]] = {"estado": "treinando"}
        elif tipo == "lote":
            c = ev["cliente"]
            self.cartoes[c] = {"estado": "treinando", "epoca": ev["epoca"], "lote": ev["lote"],
                               "loss": ev.get("loss"), "lotes_por_epoca": ev["lotes_por_epoca"]}
            self.ultimo_lote[c] = ev
            ocioso = self.cartoes.get(self.foco, {}).get("estado") != "treinando"
            if self.seguir_auto and c != self.foco and ocioso:
                self._focar(c, manual=False)
            if c == self.foco:
                # fica no máximo 1 lote esperando: a animação nunca "fica para trás"
                pendentes = [e for e in self.fila if e["tipo"] == "lote"]
                if len(pendentes) >= 1:
                    self.fila.remove(pendentes[0])
                self.fila.append({"tipo": "lote", "fases": FASES_LOTE, "ev": ev,
                                  "intervalo": f"{ev.get('intervalo', ev['lotes_por_epoca'])} lotes"})
        elif tipo == "envio":
            c = ev["cliente"]
            self.cartoes[c] = {"estado": "enviou"}
            if hasattr(self, "pos_clientes") and c < len(self.pos_clientes):
                self._voar(self.pos_clientes[c], self.pos_servidor, COR["positivo"])
            if c == self.foco:
                self.fila.append({"tipo": "envio", "fases": [("envio", 0.6)], "ev": ev})
        elif tipo == "agregado":
            self.fila.append({"tipo": "agregado", "fases": [("fedavg", 1.3)], "ev": ev})
        elif tipo == "avaliacao":
            self.metricas = [m for m in self.metricas if m[0] != ev["rodada"]]
            self.metricas.append((ev["rodada"], ev.get("accuracy"), ev.get("f1"), ev.get("loss")))
            self.metricas.sort()
            self._desenhar_metricas()
            self.fila.append({"tipo": "avaliacao", "fases": [("avaliacao", 0.8)], "ev": ev})
        elif tipo == "fim":
            self.concluido = True
            self.fila.append({"tipo": "fim", "fases": [("fim", 0.1)], "ev": ev})
        self._atualizar_cartoes()

    # --------------------------------------------- motor de animação ---
    def _tick(self):
        agora = time.time()
        dt = agora - self.ultimo_tick
        self.ultimo_tick = agora
        try:
            while True:
                self._processar(self.eventos.get_nowait())
        except queue.Empty:
            pass

        if self.atual and self.pausado:
            self.atual["t0"] += dt          # congela a animação
        if self.atual is None and self.fila and not self.pausado:
            self._iniciar(self.fila.popleft())
        if self.atual:
            # Se há fila acumulada, acelera para alcançar o treino.
            acel = self.velocidade if self.velocidade < 1000 else 12.0
            acel *= 1 + 0.5 * len(self.fila)
            dec = (agora - self.atual["t0"]) * acel
            acumulado = 0.0
            for nome, dur in self.atual["fases"]:
                if dec < acumulado + dur:
                    self._quadro(nome, (dec - acumulado) / dur)
                    break
                acumulado += dur
            else:
                self._finalizar()
        self._animar_voos()
        self._status()
        self.raiz.after(33, self._tick)

    def _iniciar(self, item):
        self.atual = item
        item["t0"] = time.time()
        self.fase_atual = None
        tipo, ev = item["tipo"], item.get("ev", {})
        if tipo == "lote":
            self.pesos_antes = self.pesos_vistos or ev["pesos"]
            self._titulo_rede(f"Rede local do cliente {ev['cliente']} · round {ev['rodada']}")
            for l, it in enumerate(self.it_rot_entrada):
                v = ev["ativacoes"][0][l]
                nome = self.cfg["colunas"][l] if l < len(self.cfg["colunas"]) else f"x{l}"
                self.cv_rede.itemconfigure(it, text=f"{nome} {v:+.2f}")
        elif tipo == "rodada":
            self.pesos_vistos = ev["pesos"]
            self._pintar_pesos(ev["pesos"])
            self._limpar_neuronios()
            self._esconder_pontos()
            self._titulo_rede(f"Modelo global enviado no round {ev['rodada']}")
            self.cv_rede.itemconfigure(self.it_saida, text="")
            if hasattr(self, "pos_clientes"):
                for k, pos in enumerate(self.pos_clientes):
                    self._voar(self.pos_servidor, pos, COR["global"], 0.8, atraso=0.05 * k)
        elif tipo == "agregado":
            self.pesos_antes = self.pesos_vistos or ev["pesos"]
            self._limpar_neuronios()
            self._esconder_pontos()
            self._titulo_rede(f"Modelo global após FedAvg · round {ev['rodada']}")
            self.cv_rede.itemconfigure(self.it_saida, text="")

    def _finalizar(self):
        item = self.atual
        if item["tipo"] in ("lote", "agregado"):
            self.pesos_vistos = item["ev"]["pesos"]
            self._pintar_pesos(self.pesos_vistos)
        self._esconder_pontos()
        if item["tipo"] == "agregado" and hasattr(self, "it_servidor"):
            self.cv_fed.itemconfigure(self.it_servidor, fill=COR["fundo"], width=2)
            self.cv_fed.itemconfigure(self.it_formula, fill=COR["texto3"])
        self.atual = None

    def _quadro(self, fase, p):
        item = self.atual
        if fase != self.fase_atual:
            self.fase_atual = fase
            self._destacar_codigo(fase)
            self._explicar(fase, self._texto_fase(fase, item))
            self._entrar_fase(fase, item)
        if item["tipo"] == "lote":
            self._quadro_lote(fase, p, item["ev"])
        elif item["tipo"] == "agregado":
            self._quadro_fedavg(p, item["ev"])

    def _entrar_fase(self, fase, item):
        if item["tipo"] != "lote":
            return
        ev = item["ev"]
        if fase == "entrada":
            self._limpar_neuronios()
            self._pintar_pesos(self.pesos_antes, forca=0.45)
            self.cv_rede.itemconfigure(self.it_saida, text="")
        elif fase.startswith("camada"):
            l = int(fase[-1])
            if l > 0:   # a camada anterior volta ao "normal"
                self._pintar_camada_pesos(l - 1, self.pesos_antes[l - 1], 0.45)
        elif fase == "loss":
            self._pintar_camada_pesos(len(self.it_arestas) - 1, self.pesos_antes[-1], 0.45)
            self._esconder_pontos()
        elif fase.startswith("backward"):
            l = int(fase[-1])
            if l + 1 < len(self.it_arestas):
                self._pintar_camada_gradiente(l + 1, ev["gradientes"][l + 1], 0.35)
        elif fase == "step":
            self._esconder_pontos()

    def _pintar_camada_pesos(self, l, W, forca):
        esc = maximo_abs(W)
        for j, linha in enumerate(W):
            for i, v in enumerate(linha):
                cor, larg = self._cor_peso(v, esc, forca)
                self.cv_rede.itemconfigure(self.it_arestas[l][j][i], fill=cor, width=larg)

    def _pintar_camada_gradiente(self, l, G, forca):
        if G is None:
            return
        esc = maximo_abs(G)
        for j, linha in enumerate(G):
            for i, g in enumerate(linha):
                t = min(1.0, abs(g) / esc) * forca
                self.cv_rede.itemconfigure(self.it_arestas[l][j][i],
                                           fill=misturar(COR["painel"], COR["gradiente"], 0.1 + 0.9 * t),
                                           width=0.5 + 3.0 * t)

    def _quadro_lote(self, fase, p, ev):
        a = ev["ativacoes"]
        W = self.pesos_antes
        if fase == "entrada":
            self._pintar_neuronios(0, a[0], suave(p))
        elif fase.startswith("camada"):
            l = int(fase[-1])
            # contribuição de cada aresta: |a_i · w_ji|
            contrib = [[abs(a[l][i] * W[l][j][i]) for i in range(len(a[l]))] for j in range(len(a[l + 1]))]
            esc = maximo_abs(contrib)
            brilho = math.sin(math.pi * min(1, p * 1.15)) if p < 0.87 else 0.35 + 0.65 * (1 - p) / 0.13
            for j, linha in enumerate(contrib):
                for i, c in enumerate(linha):
                    t = c / esc
                    cor = misturar(COR["painel"], COR["ativo"], (0.08 + 0.92 * t) * (0.4 + 0.6 * brilho))
                    self.cv_rede.itemconfigure(self.it_arestas[l][j][i], fill=cor, width=0.5 + 3.2 * t)
            fortes = sorted(((c, i, j) for j, linha in enumerate(contrib) for i, c in enumerate(linha)),
                            reverse=True)[:len(self.it_pontos)]
            self._pontos([(self.pos[l][i], self.pos[l + 1][j]) for _, i, j in fortes], suave(p), COR["ativo"])
            escala = max([abs(v) for v in a[l + 1]] + [1e-6])
            self._pintar_neuronios(l + 1, a[l + 1], suave(p), escala)
            if l + 1 == len(a) - 1:
                self.cv_rede.itemconfigure(self.it_saida, text=f"P(inadimplente)\n= {a[-1][0]:.3f}")
        elif fase == "loss":
            y = ev.get("rotulo")
            pr = a[-1][0]
            pisca = 0.5 + 0.5 * math.sin(p * math.pi * 4)
            erro = abs((y or 0) - pr)
            cor = misturar(COR["ok"], COR["gradiente"], erro)
            for it in self.it_nos[-1]:
                self.cv_rede.itemconfigure(it, outline=misturar(COR["painel"], cor, 0.4 + 0.6 * pisca), width=4)
            real = "1 (inadimplente)" if y == 1 else "0 (adimplente)"
            loss = ev.get("loss")
            self.cv_rede.itemconfigure(
                self.it_saida,
                text=f"P(inadimplente) = {pr:.3f}\nreal = {real}\nloss do lote = {loss:.4f}" if loss is not None
                else f"P = {pr:.3f}\nreal = {real}")
        elif fase.startswith("backward"):
            l = int(fase[-1])
            for it in self.it_nos[-1]:
                self.cv_rede.itemconfigure(it, width=1.5)
            G = ev["gradientes"][l]
            if G is None:
                return
            self._pintar_camada_gradiente(l, G, 0.25 + 0.75 * suave(p))
            fortes = sorted(((abs(g), i, j) for j, linha in enumerate(G) for i, g in enumerate(linha)),
                            reverse=True)[:len(self.it_pontos)]
            self._pontos([(self.pos[l][i], self.pos[l + 1][j]) for _, i, j in fortes], suave(p),
                         COR["gradiente"], reverso=True)
        elif fase == "step":
            q = suave(p)
            novos = ev["pesos"]
            interp = [[[w0 + (w1 - w0) * q for w0, w1 in zip(l0, l1)] for l0, l1 in zip(m0, m1)]
                      for m0, m1 in zip(W, novos)]
            self._pintar_pesos(interp, forca=0.45 + 0.4 * q)

    def _quadro_fedavg(self, p, ev):
        novos = ev["pesos"]
        W = self.pesos_antes
        if p < 0.5:   # servidor "pisca" enquanto agrega
            pisca = 0.5 + 0.5 * math.sin(p * math.pi * 8)
            self.cv_fed.itemconfigure(self.it_servidor, fill=misturar(COR["fundo"], COR["global"], 0.35 * pisca),
                                      width=3)
            self.cv_fed.itemconfigure(self.it_formula, fill=COR["global"])
        else:
            q2 = suave((p - 0.5) / 0.5)
            interp = [[[w0 + (w1 - w0) * q2 for w0, w1 in zip(l0, l1)] for l0, l1 in zip(m0, m1)]
                      for m0, m1 in zip(W, novos)]
            self._pintar_pesos(interp, forca=0.75)

    # --------------------------------------------------------- status ---
    def _status(self):
        partes = []
        if self.rodada:
            partes.append(f"Round {self.rodada}/{self.cfg['rodadas']}")
        c = self.cartoes.get(self.foco, {})
        partes.append(f"seguindo o cliente {self.foco}" + ("" if self.seguir_auto else " (fixado)"))
        if "epoca" in c:
            partes.append(f"época {c['epoca'] + 1}/{self.cfg['epocas']} · lote {c['lote'] + 1}/{c['lotes_por_epoca']}")
        if self.pausado:
            partes.append("⏸ PAUSADO (o treino também espera)")
        elif self.concluido and not self.atual and not self.fila:
            partes = ["✓ treino concluído"]
        self.lbl_status.configure(text=seguro("   ·   ".join(partes)))


# ============================================================ UDP ===
def _escutar(sock, janela):
    while True:
        try:
            dados, _ = sock.recvfrom(65535)
            janela.receber(decodificar(dados))
        except (OSError, ValueError):
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codigo", required=True, help="arquivo de código a exibir")
    ap.add_argument("--controle", default=None, help="arquivo JSON de controle (pausa/velocidade)")
    ap.add_argument("--porta", type=int, default=0, help="porta UDP (0 = escolher uma livre)")
    args = ap.parse_args()

    raiz = tk.Tk()
    global _SEM_UNICODE
    try:
        _SEM_UNICODE = raiz.tk.call("::tk::pkgconfig", "get", "fontsystem") != "xft"
    except tk.TclError:
        _SEM_UNICODE = False
    controle = Controle(args.controle)
    janela = Janela(raiz, args.codigo, controle)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind((HOST, args.porta))
    threading.Thread(target=_escutar, args=(sock, janela), daemon=True).start()

    def fechar():
        controle.apagar()      # o treino percebe e segue em velocidade máxima
        raiz.destroy()

    raiz.protocol("WM_DELETE_WINDOW", fechar)
    raiz.update()
    print(f"PORTA {sock.getsockname()[1]}", flush=True)   # avisa quem abriu a janela
    try:
        raiz.mainloop()
    except KeyboardInterrupt:
        fechar()


if __name__ == "__main__":
    main()
