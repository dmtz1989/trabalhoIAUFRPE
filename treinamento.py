"""
treinamento.py - Prepara os dados, treina a MLP e salva o modelo.

Uso:  python treinamento.py
"""
import os
import numpy as np
import pandas as pd
from mlp import MLP

ARQUIVO_DADOS = "default of credit card clients.xls"
ARQUIVO_MODELO = "modelo_mlp.npz"
COLUNA_ALVO = "default payment next month"

# As 4 entradas escolhidas (ver explicação no final do README / apresentação)
NOMES_ENTRADAS = [
    "PAY_0",       # situação do pagamento no mês mais recente
    "PAY_MAX",     # pior atraso nos últimos 6 meses
    "LOG_LIMITE",  # log do limite de crédito (perfil financeiro)
    "LOG_PAGTO",   # log do valor pago na última fatura
]
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


def montar_entradas(historico_pay, limite, ultimo_pagamento):
    """
    Transforma os dados "crus" de UM cliente nas 4 entradas da rede.
    É usada tanto no treino quanto no cadastro, garantindo a MESMA transformação.

    historico_pay: lista com 6 status (do mês mais recente para o mais antigo)
                   -2 = sem consumo | -1 = pago em dia | 0 = pagamento mínimo
                   1..9 = meses de atraso
    """
    return [
        historico_pay[0],
        max(historico_pay),
        np.log1p(limite),
        np.log1p(ultimo_pagamento),
    ]


def ler_planilha(caminho):
    """Lê o banco (.xls, .xlsx ou .csv) e padroniza os nomes das colunas."""
    ext = os.path.splitext(caminho)[1].lower()
    leitor = pd.read_csv if ext == ".csv" else pd.read_excel
    df = None
    for cabecalho in (1, 0):     # o .xls original tem uma linha extra antes do cabeçalho
        df = leitor(caminho, header=cabecalho)
        if "LIMIT_BAL" in df.columns:
            break
    if "PAY_0" not in df.columns and "PAY_1" in df.columns:   # variação do dataset
        df = df.rename(columns={"PAY_1": "PAY_0"})
    for c in df.columns:                                       # variação: default.payment.next.month
        if str(c).lower().startswith("default"):
            df = df.rename(columns={c: COLUNA_ALVO})
    faltando = [c for c in ["LIMIT_BAL", "PAY_AMT1"] + PAY_COLS if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes no arquivo: {faltando}")
    return df


def montar_entradas_df(df):
    """Versão em lote (vetorizada) de montar_entradas: todas as linhas de uma vez."""
    return np.column_stack([
        df["PAY_0"].values,
        df[PAY_COLS].max(axis=1).values,
        np.log1p(df["LIMIT_BAL"].values),
        np.log1p(df["PAY_AMT1"].values),
    ]).astype(float)


def carregar_dados(caminho=ARQUIVO_DADOS):
    df = ler_planilha(caminho)
    X = montar_entradas_df(df)
    y = df[COLUNA_ALVO].values.reshape(-1, 1).astype(float)
    return X, y


def dividir(n, semente=42):
    """70% treino | 15% validação | 15% teste"""
    idx = np.random.default_rng(semente).permutation(n)
    a, b = int(0.70 * n), int(0.85 * n)
    return idx[:a], idx[a:b], idx[b:]


# ---------- Métricas ----------
def auc_roc(scores, y):
    y, s = y.ravel(), scores.ravel()
    ranks = pd.Series(s).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def metricas(y_real, y_prev):
    y_real, y_prev = y_real.ravel(), y_prev.ravel()
    vp = int(((y_prev == 1) & (y_real == 1)).sum())
    fp = int(((y_prev == 1) & (y_real == 0)).sum())
    vn = int(((y_prev == 0) & (y_real == 0)).sum())
    fn = int(((y_prev == 0) & (y_real == 1)).sum())
    prec = vp / (vp + fp) if vp + fp else 0
    rec = vp / (vp + fn) if vp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return dict(acuracia=(vp + vn) / len(y_real), precisao=prec, recall=rec,
                f1=f1, vp=vp, fp=fp, vn=vn, fn=fn)


def melhor_limiar(probs, y):
    """Escolhe o limiar de decisão que maximiza o F1 (na validação)."""
    melhor = (0, 0.5)
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = metricas(y, (probs >= t).astype(int))["f1"]
        if f1 > melhor[0]:
            melhor = (f1, t)
    return float(melhor[1])


def treinar_e_salvar():
    """Treina a rede, avalia no teste e salva o modelo em modelo_mlp.npz."""
    print("Carregando dados...")
    X, y = carregar_dados()
    tr, va, te = dividir(len(X))

    # Normalização (média 0, desvio 1) usando APENAS estatísticas do treino
    media, desvio = X[tr].mean(axis=0), X[tr].std(axis=0)
    Xn = (X - media) / desvio

    print(f"Treino: {len(tr)} | Validação: {len(va)} | Teste: {len(te)}")
    print(f"Proporção de inadimplentes: {y.mean():.1%}\n")

    rede = MLP(n_entradas=4, n_ocultos=8, taxa_aprendizado=0.5)
    hist = rede.treinar(Xn[tr], y[tr], Xn[va], y[va], epocas=100, tamanho_lote=64)

    # Limiar de decisão: como só ~22% são inadimplentes, 0.5 quase nunca dispara
    limiar = melhor_limiar(rede.prever_proba(Xn[va]), y[va])

    print("\n===== RESULTADO NO CONJUNTO DE TESTE (dados nunca vistos) =====")
    p_te = rede.prever_proba(Xn[te])
    m = metricas(y[te], (p_te >= limiar).astype(int))
    print(f"Limiar de decisão : {limiar:.2f}")
    print(f"AUC-ROC           : {auc_roc(p_te, y[te]):.3f}")
    print(f"Acurácia          : {m['acuracia']:.1%}  (chutar 'sempre paga' dá {1 - y[te].mean():.1%})")
    print(f"Precisão          : {m['precisao']:.1%}")
    print(f"Recall            : {m['recall']:.1%}")
    print(f"F1                : {m['f1']:.3f}")
    print(f"Matriz de confusão: VP={m['vp']} FP={m['fp']} VN={m['vn']} FN={m['fn']}")

    rede.salvar(ARQUIVO_MODELO, media=media, desvio=desvio, limiar=limiar)
    print(f"\nModelo salvo em '{ARQUIVO_MODELO}'")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(hist["erro_treino"], label="Treino")
        plt.plot(hist["erro_val"], label="Validação")
        plt.xlabel("Época"); plt.ylabel("MSE"); plt.title("Curva de aprendizado")
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig("curva_aprendizado.png", dpi=120)
        print("Gráfico salvo em 'curva_aprendizado.png'")
    except ImportError:
        pass


if __name__ == "__main__":
    treinar_e_salvar()
