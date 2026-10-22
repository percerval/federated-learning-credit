# 🔬 Visualizador ao vivo do Aprendizado Federado

Roda a **mesma simulação** de `machine_learning.py` no terminal e, ao mesmo tempo, abre uma
**janela flutuante** que mostra, sincronizados:

| Painel | O que aparece |
|---|---|
| **Código** | `machine_learning.py` original, com a linha que acabou de executar destacada (e a função em volta) |
| **Rede neural** | a MLP 12 → 16 → 8 → 1 animada: entrada do lote, forward camada a camada, loss, backpropagation e atualização dos pesos |
| **Federação** | servidor e clientes: o modelo global descendo, os pesos locais subindo e o FedAvg |
| **Métricas** | acurácia, F1 e loss globais a cada round |
| **Explicação** | o que está acontecendo agora, com os números reais (P(inadimplente), rótulo, loss, gradientes, pesos do FedAvg) |

## ▶️ Como executar

```bash
# Tkinter (interface gráfica) — uma vez só
sudo apt install python3-tk

# Na raiz do repositório
python3 -m visualizador
```

Opções:

```bash
python3 -m visualizador --amostrar 10   # anima 1 a cada 10 lotes (padrão: 1 por época)
python3 -m visualizador --sem-janela    # só terminal, para comparar com o original
```

O terminal continua mostrando a mesma saída do script original. Ao fim do treino a janela
fica aberta para você explorar.

## 🎛️ Controles da janela

- **Pausar / Continuar** — pausa a animação **e o treino** (os clientes esperam), ótimo para explicar em aula.
- **◀◀ / ▶▶** — velocidade de 0.25x a 8x, ou **máx** (treino sem esperas; a animação pula para o lote mais recente).
- **A− / A+** — tamanho da fonte do código (útil no projetor).
- **Sempre no topo** — mantém a janela flutuando sobre o terminal.
- **Clique em um cliente** (C0, C1, C2) para fixar a rede daquele cliente. Clique de novo para voltar ao
  modo automático, que segue sempre quem está treinando.

## 🧠 Como funciona

`machine_learning.py` **não foi alterado** (os testes do repositório conferem isso byte a byte).
O visualizador importa o código original e o observa de fora:

```
python3 -m visualizador
│
├── abre a janela (processo separado, Tkinter)
│
└── simulação Flower (espelho do main() original)
    ├── servidor: FedAvgVisual  ── herda de FedAvg, avisa: round, recebeu, agregou, avaliou
    └── clientes: ClienteVisual ── herda de ClienteHMEQ; durante o fit() instala hooks do PyTorch
                                   e chama o treinar() ORIGINAL
                                        │
            hooks: entrada de cada camada Linear → ativações
                   saída do modelo               → P(inadimplente)
                   BCELoss                        → loss e rótulo
                   otimizador.step()              → gradientes e pesos novos
                                        │
                     eventos JSON via UDP (127.0.0.1) ──▶ janela
                     arquivo de controle (pausa/velocidade) ◀── janela
```

- **UDP** porque os clientes rodam em processos do Ray: enviar um evento nunca trava o treino,
  mesmo que a janela seja fechada no meio (aí o treino segue em velocidade máxima).
- Por padrão é animado **1 lote por época** de cada cliente; o treino espera o tempo da animação
  para que o que você vê seja o que está rodando agora. Os demais lotes treinam normalmente.
- Os pesos mostrados depois do `otimizador.step()` já incluem todas as atualizações feitas desde o
  último lote animado.

## 📁 Arquivos

```
visualizador/
├── __main__.py     # python3 -m visualizador (argumentos)
├── simulacao.py    # ClienteVisual, FedAvgVisual e o main() espelhado
├── observador.py   # hooks do PyTorch que "fotografam" um lote
├── protocolo.py    # eventos UDP + arquivo de controle
└── janela.py       # a janela Tkinter (código, rede, federação, métricas)
tests/test_visualizador.py   # inclusive: treino com e sem observador dá pesos idênticos
```

## ❓ Problemas comuns

| Sintoma | Solução |
|---|---|
| `Não foi possível abrir a janela (Tkinter instalado?)` | `sudo apt install python3-tk` |
| `Nenhum ambiente gráfico encontrado` | rodando via SSH sem X: use `ssh -X` ou rode localmente |
| Treino muito lento para a apresentação | **▶▶** para 2x/4x, ou até `máx` |
