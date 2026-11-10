"""
validacoes.py - Bateria de testes que comprova o funcionamento do projeto.

Uso:  python validacoes.py
Requer o banco de dados (.xls) na pasta. Não altera o modelo nem os clientes
salvos: os testes que gravam arquivos usam uma pasta temporária.
"""
import contextlib
import io
import os
import sys
import tempfile
import time
from unittest import mock

import numpy as np

import app
from mlp import MLP
from treinamento import (ARQUIVO_DADOS, ARQUIVO_MODELO, COLUNA_ALVO, PAY_COLS,
                         auc_roc, carregar_dados, dividir, ler_planilha, metricas,
                         montar_entradas, montar_entradas_df, treinar_e_salvar)

PASTA = os.path.dirname(os.path.abspath(__file__))
resultados = []


def registrar(nome, ok, detalhe=""):
    resultados.append(ok)
    print(f"[{'OK' if ok else 'FALHOU'}] {nome}" + (f"  ->  {detalhe}" if detalhe else ""))


def silencio():
    return contextlib.redirect_stdout(io.StringIO())


def ler_npz(caminho):
    """Lê um .npz para um dicionário e fecha o arquivo (necessário no Windows)."""
    with np.load(caminho) as d:
        return {k: d[k] for k in d.files}


# ============ A) A REDE NEURAL ============
def teste_gradiente():
    """Compara o gradiente do backpropagation com o gradiente numérico."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    y = (rng.random((20, 1)) > 0.5).astype(float)
    rede = MLP(4, 8, taxa_aprendizado=1.0, semente=1)   # lr = 1 -> variação do peso = -gradiente
    rede.b1 = rng.normal(size=(1, 8)) * 0.1
    rede.b2 = rng.normal(size=(1, 1)) * 0.1
    nomes = ["W1", "b1", "W2", "b2"]
    antes = {n: getattr(rede, n).copy() for n in nomes}

    rede.forward(X); rede.backward(y)
    analitico = {n: antes[n] - getattr(rede, n) for n in nomes}
    for n in nomes:
        setattr(rede, n, antes[n].copy())

    eps, pior = 1e-6, 0.0
    for n in nomes:
        p = getattr(rede, n)
        numerico = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            orig = p[idx]
            p[idx] = orig + eps; perda_mais = rede.mse(rede.forward(X), y)
            p[idx] = orig - eps; perda_menos = rede.mse(rede.forward(X), y)
            p[idx] = orig
            numerico[idx] = (perda_mais - perda_menos) / (2 * eps)
        erro = np.linalg.norm(numerico - analitico[n]) / (np.linalg.norm(numerico) + np.linalg.norm(analitico[n]))
        pior = max(pior, erro)
    registrar("Backpropagation confere com o gradiente numérico (W1, b1, W2, b2)",
              pior < 1e-5, f"erro relativo máximo = {pior:.2e}")


def teste_saida_sigmoide():
    rede = MLP(4, 8)
    X = np.random.default_rng(1).normal(scale=50, size=(1000, 4))   # entradas extremas
    s = rede.prever_proba(X)
    registrar("Saída da rede sempre entre 0 e 1, com formato (n, 1)",
              s.shape == (1000, 1) and s.min() >= 0 and s.max() <= 1 and np.isfinite(s).all(),
              f"mín={s.min():.4f} máx={s.max():.4f}")


def teste_xor():
    """A rede 2-8-1 deve aprender o XOR (problema que um neurônio sozinho não resolve)."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    rede = MLP(2, 8, taxa_aprendizado=0.5, semente=3)
    erro_ini = rede.mse(rede.prever_proba(X), y)
    rede.treinar(X, y, epocas=8000, tamanho_lote=4, verbose=False)
    erro_fim = rede.mse(rede.prever_proba(X), y)
    acertos = int((rede.prever(X) == y).sum())
    registrar("Rede aprende o XOR (não linear) com ReLU + sigmoide + MSE",
              acertos == 4 and erro_fim < erro_ini,
              f"MSE {erro_ini:.3f} -> {erro_fim:.4f} | acertos {acertos}/4")


def teste_metricas():
    m = metricas(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 1]))
    ok_m = (m["vp"], m["fp"], m["vn"], m["fn"]) == (1, 1, 1, 1) and abs(m["f1"] - 0.5) < 1e-12
    y = np.array([1, 1, 0, 0])
    ok_auc = (abs(auc_roc(np.array([.9, .8, .2, .1]), y) - 1.0) < 1e-12 and
              abs(auc_roc(np.array([.1, .2, .8, .9]), y) - 0.0) < 1e-12)
    registrar("Métricas (acurácia/precisão/recall/F1/matriz) e AUC corretas em casos conhecidos",
              ok_m and ok_auc)


# ============ B) OS DADOS ============
def teste_dados():
    df = ler_planilha(os.path.join(PASTA, ARQUIVO_DADOS))
    obrig = ["LIMIT_BAL", "PAY_AMT1", COLUNA_ALVO] + PAY_COLS
    sem_nulos = not df[obrig].isna().any().any()
    alvo_ok = set(df[COLUNA_ALVO].unique()) <= {0, 1}
    pay_min, pay_max = int(df[PAY_COLS].min().min()), int(df[PAY_COLS].max().max())
    registrar("Banco íntegro: 30.000 linhas, IDs únicos, sem nulos, alvo binário",
              len(df) == 30000 and df["ID"].is_unique and sem_nulos and alvo_ok,
              f"inadimplentes={df[COLUNA_ALVO].mean():.1%} | PAY de {pay_min} a {pay_max} | "
              f"limite de {int(df.LIMIT_BAL.min()):,} a {int(df.LIMIT_BAL.max()):,}")
    registrar("Valores de pagamento nunca negativos", bool((df["PAY_AMT1"] >= 0).all()))


def teste_entradas_lote_vs_individual():
    df = ler_planilha(os.path.join(PASTA, ARQUIVO_DADOS)).iloc[:5000]
    lote = montar_entradas_df(df)
    indiv = np.array([montar_entradas(l[PAY_COLS].tolist(), l["LIMIT_BAL"], l["PAY_AMT1"])
                      for _, l in df.iterrows()])
    registrar("Entradas em lote (importação) idênticas às do cadastro individual",
              np.allclose(lote, indiv), "5.000 clientes comparados")


def teste_divisao_e_normalizacao():
    X, y = carregar_dados(os.path.join(PASTA, ARQUIVO_DADOS))
    tr, va, te = dividir(len(X))
    sem_sobreposicao = not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    registrar("Divisão 70/15/15 sem sobreposição entre treino, validação e teste",
              (len(tr), len(va), len(te)) == (21000, 4500, 4500) and sem_sobreposicao
              and len(set(tr) | set(va) | set(te)) == len(X),
              f"{len(tr)}/{len(va)}/{len(te)}")
    _, params = MLP.carregar(os.path.join(PASTA, ARQUIVO_MODELO))
    registrar("Normalização do modelo usa somente estatísticas do treino (sem vazamento)",
              np.allclose(params["media"], X[tr].mean(0)) and np.allclose(params["desvio"], X[tr].std(0)))


# ============ C) O MODELO SALVO ============
def teste_modelo_salvo():
    rede, params = MLP.carregar(os.path.join(PASTA, ARQUIVO_MODELO))
    registrar("Modelo salvo tem a arquitetura 4-8-1 e limiar válido",
              rede.W1.shape == (4, 8) and rede.W2.shape == (8, 1) and 0 < float(params["limiar"]) < 1,
              f"limiar = {float(params['limiar']):.2f}")


def teste_reprodutibilidade():
    """Retreina do zero numa pasta temporária e compara com o modelo salvo."""
    original = ler_npz(os.path.join(PASTA, ARQUIVO_MODELO))
    atual = os.getcwd()
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(PASTA, ARQUIVO_DADOS), "rb") as f_in, \
                open(os.path.join(tmp, ARQUIVO_DADOS), "wb") as f_out:
            f_out.write(f_in.read())
        os.chdir(tmp)
        try:
            with silencio():
                treinar_e_salvar()
            novo = ler_npz(os.path.join(tmp, ARQUIVO_MODELO))
            iguais = all(np.allclose(original[k], novo[k]) for k in
                         ["W1", "b1", "W2", "b2", "media", "desvio", "limiar"])
        finally:
            os.chdir(atual)
    registrar("Retreinar do zero gera exatamente o mesmo modelo (sementes fixas)",
              iguais, f"treino refeito em {time.time() - t0:.0f}s")


# ============ D) A APLICAÇÃO ============
def teste_validacao_de_entrada():
    with mock.patch("builtins.input", side_effect=["abc", "-5", "0", "2000"]), silencio():
        v1 = app.ler_numero("", minimo=1)
    with mock.patch("builtins.input", side_effect=["1.5", "10", "-3", "3"]), silencio():
        v2 = app.ler_numero("", -2, 9, inteiro=True)
    with mock.patch("builtins.input", side_effect=["1,5"]), silencio():
        v3 = app.ler_numero("", minimo=0)
    registrar("Cadastro rejeita texto, fora da faixa e decimal onde exige inteiro; aceita vírgula",
              (v1, v2, v3) == (2000, 3, 1.5), f"aceitos: {v1}, {v2}, {v3}")


def teste_importacao():
    rede, params = MLP.carregar(os.path.join(PASTA, ARQUIVO_MODELO))
    arquivo = os.path.join(PASTA, ARQUIVO_DADOS)
    with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(app, "ARQUIVO_CLIENTES", os.path.join(tmp, "clientes.json")):
        with mock.patch("builtins.input", return_value=arquivo), silencio():
            app.importar_banco(rede, params)
        n1 = len(app.carregar_clientes())

        saida = io.StringIO()
        with mock.patch("builtins.input", return_value=arquivo), contextlib.redirect_stdout(saida):
            app.importar_banco(rede, params)
        n2 = len(app.carregar_clientes())
        registrar("Importação traz os 30.000 clientes e não duplica ao importar de novo",
                  n1 == 30000 and n2 == 30000 and "já estavam" in saida.getvalue(),
                  f"1ª importação: {n1} | 2ª: {n2}")

        clientes = app.carregar_clientes()
        passo = len(clientes) // 300
        dif, flags_ok = 0.0, True
        for c in clientes[::passo]:
            prob, alto = app.prever_cliente(rede, params, c["historico_pay"], c["limite"], c["ultimo_pagamento"])
            dif = max(dif, abs(prob - c["risco"]))
            flags_ok &= (alto == c["previsao_inadimplente"])
        registrar("Previsão da importação em lote = previsão do cadastro individual",
                  dif < 1e-4 and flags_ok, f"300 clientes | maior diferença = {dif:.1e}")

        csv_teste = os.path.join(tmp, "resultados_teste.csv")
        with mock.patch.object(app, "ARQUIVO_RESULTADOS", csv_teste), silencio():
            app.exportar()
        csv_ok = os.path.exists(csv_teste)
        linhas = 0
        if csv_ok:
            with open(csv_teste, encoding="utf-8-sig") as f:
                linhas = sum(1 for _ in f)
        registrar("Exportação gera CSV com cabeçalho + 30.000 linhas", csv_ok and linhas == 30001)


def teste_arquivos_ausentes():
    with tempfile.TemporaryDirectory() as tmp:
        atual = os.getcwd()
        os.chdir(tmp)       # pasta vazia: sem modelo e sem banco
        try:
            saida = io.StringIO()
            with mock.patch("builtins.input", side_effect=["0"]), contextlib.redirect_stdout(saida):
                app.main()
        finally:
            os.chdir(atual)
    registrar("Sem modelo e sem banco, o app avisa em vez de quebrar",
              "também não está na pasta" in saida.getvalue())


def main():
    inicio = time.time()
    os.chdir(PASTA)
    for bloco, testes in [
        ("A) A rede neural", [teste_gradiente, teste_saida_sigmoide, teste_xor, teste_metricas]),
        ("B) Os dados", [teste_dados, teste_entradas_lote_vs_individual, teste_divisao_e_normalizacao]),
        ("C) O modelo salvo", [teste_modelo_salvo, teste_reprodutibilidade]),
        ("D) A aplicação", [teste_validacao_de_entrada, teste_importacao, teste_arquivos_ausentes]),
    ]:
        print(f"\n--- {bloco} ---")
        for t in testes:
            t()
    ok = sum(resultados)
    print(f"\n{ok}/{len(resultados)} validações passaram em {time.time() - inicio:.0f}s")
    sys.exit(0 if ok == len(resultados) else 1)


if __name__ == "__main__":
    main()
