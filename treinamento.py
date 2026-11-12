"""
treinamento.py - Prepara os dados, treina a MLP e salva o modelo.

Protocolo:
  1. Divisão estratificada: 70% da classe 0 + 70% da classe 1 (treino); 30% de cada (teste)
  2. Oversampling da classe 1 SOMENTE no treino, até igualar a classe 0
  3. Treino da MLP (4 entradas, 8 neurônios ReLU, 1 saída sigmoide, MSE, backpropagation)
  4. Teste nos 30% restantes, com matriz de confusão (VP, FP, VN, FN)

Uso:  python treinamento.py
"""
import os
import numpy as np
import pandas as pd
from mlp import MLP

ARQUIVO_DADOS = "default_of_credit_card_clients.xls"
ARQUIVO_MODELO = "modelo_mlp.npz"
COLUNA_ALVO = "default payment next month"

PROPORCAO_TREINO = 0.70   # 70% de cada classe para treino, 30% de cada classe para teste
SEMENTE = 42
LIMIAR = 0.5              # com o treino balanceado, 0,5 é o ponto de corte natural
TAXA_APRENDIZADO = 0.05   # menor que antes: com treino balanceado, 0,5 deixava a curva instável

# As 4 entradas escolhidas (a justificativa está no README, seção 'Entradas e saída da rede')
NOMES_ENTRADAS = [
    "PAY_0",       # situação do pagamento no mês mais recente
    "PAY_MAX",     # pior atraso nos últimos 6 meses
    "LOG_LIMITE",  # log do limite de crédito (perfil financeiro)
    "LOG_PAGTO",   # log do valor pago na última fatura
]
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


def _ler_tabela(caminho, cabecalho):
    """Lê .csv ou planilha Excel. Para Excel, prefere o leitor 'calamine' (rápido e mantido);
    se ele não estiver instalado, usa o padrão do pandas (xlrd para .xls, openpyxl para .xlsx)."""
    if os.path.splitext(caminho)[1].lower() == ".csv":
        return pd.read_csv(caminho, header=cabecalho)
    try:
        return pd.read_excel(caminho, header=cabecalho, engine="calamine")
    except (ImportError, ValueError):        # calamine ausente ou pandas antigo
        return pd.read_excel(caminho, header=cabecalho)


def ler_planilha(caminho):
    """Lê o banco (.xls, .xlsx ou .csv) e padroniza os nomes das colunas."""
    df = None
    for cabecalho in (1, 0):     # o .xls original tem uma linha extra antes do cabeçalho
        df = _ler_tabela(caminho, cabecalho)
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


def montar_entradas(df):
    """
    Transforma os dados "crus" dos clientes (um DataFrame, uma linha por cliente) nas 4
    entradas da rede. É usada no treino, na importação e no cadastro individual, garantindo
    a MESMA transformação nos três casos.

    PAY_*: -2 = sem consumo | -1 = pago em dia | 0 = pagamento mínimo | 1..9 = meses de atraso
    """
    return np.column_stack([
        df["PAY_0"].values,
        df[PAY_COLS].max(axis=1).values,
        np.log1p(df["LIMIT_BAL"].values),
        np.log1p(df["PAY_AMT1"].values),
    ]).astype(float)


def carregar_dados(caminho=ARQUIVO_DADOS):
    df = ler_planilha(caminho)
    X = montar_entradas(df)
    y = df[COLUNA_ALVO].values.reshape(-1, 1).astype(float)
    return X, y


def dividir_estratificado(y, proporcao_treino=PROPORCAO_TREINO, semente=SEMENTE):
    """
    Divisão ESTRATIFICADA: 70% dos exemplos da classe 0 + 70% dos exemplos da classe 1
    vão para o treino; os 30% restantes de CADA classe vão para o teste.
    Assim treino e teste mantêm a mesma proporção de inadimplentes do banco original.
    """
    rotulos = y.ravel()
    rng = np.random.default_rng(semente)
    treino, teste = [], []
    for classe in (0, 1):
        idx = rng.permutation(np.where(rotulos == classe)[0])
        corte = int(round(proporcao_treino * len(idx)))
        treino.append(idx[:corte])
        teste.append(idx[corte:])
    return rng.permutation(np.concatenate(treino)), rng.permutation(np.concatenate(teste))


def balancear_oversampling(idx_treino, y, semente=SEMENTE):
    """
    Oversampling da classe minoritária (inadimplentes): sorteia, com reposição,
    exemplos extras dessa classe até ela ficar com o mesmo número de exemplos da outra.
    Deve ser aplicado SOMENTE ao conjunto de treino, depois da divisão; se fosse feito
    antes, cópias dos mesmos clientes cairiam no treino e no teste (vazamento).
    """
    rotulos = y.ravel()[idx_treino]
    idx0, idx1 = idx_treino[rotulos == 0], idx_treino[rotulos == 1]
    menor, maior = (idx1, idx0) if len(idx1) < len(idx0) else (idx0, idx1)
    rng = np.random.default_rng(semente)
    extras = rng.choice(menor, size=len(maior) - len(menor), replace=True)
    return rng.permutation(np.concatenate([idx0, idx1, extras]))


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


def imprimir_matriz(m):
    """Matriz de confusão: linhas = classe real, colunas = classe prevista."""
    print("                       Previsto: paga   Previsto: calote")
    print(f"  Real: paga   (0)     VN = {m['vn']:<10}    FP = {m['fp']}")
    print(f"  Real: calote (1)     FN = {m['fn']:<10}    VP = {m['vp']}")


def treinar_e_salvar():
    """Treina a rede (treino estratificado + oversampling), testa nos 30% restantes
    e salva o modelo em modelo_mlp.npz."""
    print("Carregando dados...")
    X, y = carregar_dados()

    # 1) Divisão estratificada: 70% da classe 0 + 70% da classe 1 (treino); resto (teste)
    tr, te = dividir_estratificado(y)
    # 2) Oversampling da classe 1 SOMENTE no treino, até igualar a classe 0
    tr_bal = balancear_oversampling(tr, y)

    # Normalização (média 0, desvio 1) com as estatísticas do treino original
    media, desvio = X[tr].mean(axis=0), X[tr].std(axis=0)
    Xn = (X - media) / desvio

    print(f"\nEntradas da rede : {', '.join(NOMES_ENTRADAS)}")
    print(f"Banco original   : {len(y)} clientes | inadimplentes: {y.mean():.1%}")
    print(f"Treino (70%)     : {len(tr)} clientes | classe 0: {int((y[tr] == 0).sum())} | "
          f"classe 1: {int((y[tr] == 1).sum())}")
    print(f"Treino balanceado: {len(tr_bal)} exemplos | classe 0: {int((y[tr_bal] == 0).sum())} | "
          f"classe 1: {int((y[tr_bal] == 1).sum())}  (oversampling da classe 1)")
    print(f"Teste (30%)      : {len(te)} clientes | classe 0: {int((y[te] == 0).sum())} | "
          f"classe 1: {int((y[te] == 1).sum())}  (inadimplentes: {y[te].mean():.1%})\n")

    rede = MLP(n_entradas=len(NOMES_ENTRADAS), n_ocultos=8, taxa_aprendizado=TAXA_APRENDIZADO)
    hist = rede.treinar(Xn[tr_bal], y[tr_bal],
                        monitorar={"treino original": (Xn[tr], y[tr]), "teste": (Xn[te], y[te])},
                        epocas=100, tamanho_lote=64)

    print("\n===== RESULTADO NO CONJUNTO DE TESTE (30% restantes, nunca vistos) =====")
    p_te = rede.prever_proba(Xn[te])
    m = metricas(y[te], (p_te >= LIMIAR).astype(int))
    print(f"Limiar de decisão : {LIMIAR:.2f}")
    print(f"AUC-ROC           : {auc_roc(p_te, y[te]):.3f}")
    print(f"Acurácia          : {m['acuracia']:.1%}  (chutar 'sempre paga' dá {1 - y[te].mean():.1%})")
    print(f"Precisão          : {m['precisao']:.1%}")
    print(f"Recall            : {m['recall']:.1%}")
    print(f"F1                : {m['f1']:.3f}\n")
    print("Matriz de confusão (teste):")
    imprimir_matriz(m)

    rede.salvar(ARQUIVO_MODELO, media=media, desvio=desvio, limiar=LIMIAR)
    print(f"\nModelo salvo em '{ARQUIVO_MODELO}'")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(hist["treino"], label="Treino balanceado (o que a rede minimiza)")
        plt.plot(hist["treino original"], label="Treino (distribuição original)")
        plt.plot(hist["teste"], label="Teste")
        plt.xlabel("Época"); plt.ylabel("MSE"); plt.title("Curva de aprendizado")
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig("curva_aprendizado.png", dpi=120)
        plt.close()
        print("Gráfico salvo em 'curva_aprendizado.png'")
    except ImportError:
        pass


if __name__ == "__main__":
    treinar_e_salvar()
