"""
avaliar_modelo.py - Reavalia o modelo JÁ SALVO (sem retreinar) em todos os clientes.

Mostra o resultado no banco completo e separado em treino (70%) e teste (30%),
com a matriz de confusão (VP, FP, VN, FN) de cada parte.

Uso:  python avaliar_modelo.py
"""
import os
from mlp import MLP
from treinamento import (ARQUIVO_DADOS, ARQUIVO_MODELO, auc_roc, carregar_dados,
                         dividir_estratificado, imprimir_matriz, metricas,
                         treinar_e_salvar)


def relatorio(titulo, y, probs, limiar, com_matriz=False):
    m = metricas(y, (probs >= limiar).astype(int))
    print(f"{titulo:<26} n={len(y):>6} | acurácia {m['acuracia']:.1%} | "
          f"precisão {m['precisao']:.1%} | recall {m['recall']:.1%} | "
          f"F1 {m['f1']:.3f} | AUC {auc_roc(probs, y):.3f}")
    if com_matriz:
        imprimir_matriz(m)
        print()


def main():
    if not os.path.exists(ARQUIVO_DADOS):
        print(f"Banco de dados '{ARQUIVO_DADOS}' não encontrado na pasta.")
        return
    # Se o modelo ainda não existe, treina e salva (gera o modelo_mlp.npz)
    if not os.path.exists(ARQUIVO_MODELO):
        print(f"Modelo '{ARQUIVO_MODELO}' não encontrado. Treinando um novo...\n")
        treinar_e_salvar()
        print("\n" + "=" * 60 + "\n")
    rede, params = MLP.carregar(ARQUIVO_MODELO)
    X, y = carregar_dados()
    Xn = (X - params["media"]) / params["desvio"]
    probs = rede.prever_proba(Xn)
    limiar = float(params["limiar"])
    tr, te = dividir_estratificado(y)   # mesma divisão estratificada usada no treino

    print(f"Baseline ('ninguém dá calote'): {1 - y.mean():.1%} | limiar de decisão: {limiar:.2f}\n")
    relatorio("TODOS os clientes", y, probs, limiar, com_matriz=True)
    relatorio("  só treino (70%)", y[tr], probs[tr], limiar)
    relatorio("  só teste (30%)", y[te], probs[te], limiar, com_matriz=True)
    print("Obs.: o número que mede a generalização é o do teste (30% nunca vistos no treino).")


if __name__ == "__main__":
    main()
