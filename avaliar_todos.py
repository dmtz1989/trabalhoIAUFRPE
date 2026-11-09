"""
avaliar_todos.py - Aplica o modelo salvo em TODOS os clientes do banco.

Uso:  python avaliar_todos.py
"""
import numpy as np
from mlp import MLP
from treinamento import ARQUIVO_MODELO, carregar_dados, dividir, metricas, auc_roc


def relatorio(titulo, y, probs, limiar):
    m = metricas(y, (probs >= limiar).astype(int))
    print(f"{titulo:<22} n={len(y):>6} | acurácia {m['acuracia']:.1%} | "
          f"precisão {m['precisao']:.1%} | recall {m['recall']:.1%} | "
          f"F1 {m['f1']:.3f} | AUC {auc_roc(probs, y):.3f}")
    return m


def main():
    rede, params = MLP.carregar(ARQUIVO_MODELO)
    X, y = carregar_dados()
    Xn = (X - params["media"]) / params["desvio"]
    probs = rede.prever_proba(Xn)
    limiar = float(params["limiar"])
    tr, va, te = dividir(len(X))        # mesma divisão usada no treino

    print(f"Baseline ('ninguém dá calote'): {1 - y.mean():.1%}\n")
    for lim in (limiar, 0.5):
        print(f"--- Limiar {lim:.2f} ---")
        relatorio("TODOS os clientes", y, probs, lim)
        relatorio("  só treino (70%)", y[tr], probs[tr], lim)
        relatorio("  só validação (15%)", y[va], probs[va], lim)
        relatorio("  só teste (15%)", y[te], probs[te], lim)
        m = metricas(y, (probs >= lim).astype(int))
        print(f"  Matriz (todos): VP={m['vp']} FP={m['fp']} VN={m['vn']} FN={m['fn']}\n")


if __name__ == "__main__":
    main()
