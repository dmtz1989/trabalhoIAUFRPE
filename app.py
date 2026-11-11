"""
app.py - Análise de crédito com MLP: cadastro individual e importação do banco completo.

Uso:  python app.py      (se o modelo não existir, ele é treinado automaticamente)
"""
import csv
import json
import os
import numpy as np
import pandas as pd
from mlp import MLP
from treinamento import (ARQUIVO_DADOS, ARQUIVO_MODELO, COLUNA_ALVO, PAY_COLS,
                         auc_roc, imprimir_matriz, ler_planilha, metricas,
                         montar_entradas, treinar_e_salvar)

ARQUIVO_CLIENTES = "clientes.json"
ARQUIVO_RESULTADOS = "resultados_clientes.csv"

LEGENDA_PAY = """
  Situação do pagamento em cada mês:
    -2 = não usou o cartão     -1 = pagou a fatura inteira
     0 = pagou só o mínimo      1 a 9 = meses de atraso
"""


# ---------- Persistência ----------
def carregar_clientes():
    if os.path.exists(ARQUIVO_CLIENTES):
        with open(ARQUIVO_CLIENTES, encoding="utf-8") as f:
            return json.load(f)
    return []


def salvar_clientes(clientes):
    with open(ARQUIVO_CLIENTES, "w", encoding="utf-8") as f:
        json.dump(clientes, f, ensure_ascii=False)       # compacto: o banco tem 30 mil linhas


# ---------- Entrada de dados com validação ----------
def ler_numero(msg, minimo=None, maximo=None, inteiro=False):
    while True:
        txt = input(msg).strip().replace(",", ".")
        try:
            v = int(txt) if inteiro else float(txt)
        except ValueError:
            print("  ! Valor inválido, tente de novo.")
            continue
        if (minimo is not None and v < minimo) or (maximo is not None and v > maximo):
            print(f"  ! O valor deve estar entre {minimo} e {maximo}.")
            continue
        return v


# ---------- Previsão ----------
def normalizar(entradas, params):
    """Mesma normalização usada no treino (média e desvio salvos no modelo)."""
    return (entradas - params["media"]) / params["desvio"]


def prever_cliente(rede, params, historico_pay, limite, ultimo_pagamento):
    cliente = pd.DataFrame([dict(zip(PAY_COLS, historico_pay),
                                 LIMIT_BAL=limite, PAY_AMT1=ultimo_pagamento)])
    prob = float(rede.prever_proba(normalizar(montar_entradas(cliente), params))[0, 0])
    return prob, prob >= float(params["limiar"])


# ---------- 1) Cadastro individual ----------
def cadastrar(rede, params):
    print("\n=== CADASTRO DE CLIENTE ===")
    nome = input("Nome do cliente: ").strip() or "Sem nome"
    limite = ler_numero("Limite do cartão (NT$): ", minimo=1)
    print(LEGENDA_PAY)
    meses = ["último mês", "2 meses atrás", "3 meses atrás",
             "4 meses atrás", "5 meses atrás", "6 meses atrás"]
    hist = [ler_numero(f"  Situação {m} (-2 a 9): ", -2, 9, inteiro=True) for m in meses]
    pagto = ler_numero("Valor pago na última fatura (NT$): ", minimo=0)

    prob, inadimplente = prever_cliente(rede, params, hist, limite, pagto)

    print("\n--- RESULTADO DA ANÁLISE ---")
    print(f"Cliente: {nome}")
    print(f"Pontuação de risco (saída da rede): {prob:.1%}  (alto risco a partir de 50%)")
    if inadimplente:
        print("Previsão: ALTO RISCO - perfil com maior chance de NÃO pagar a próxima fatura.")
        print("Sugestão: crédito negado / limite reduzido / análise manual.")
    else:
        print("Previsão: BAIXO RISCO - perfil com maior chance de pagar a próxima fatura.")
        print("Sugestão: crédito aprovado.")
    print("Obs.: a pontuação serve para ordenar clientes por risco; não é a probabilidade real de calote.")

    clientes = carregar_clientes()
    clientes.append({"id": None, "nome": nome, "limite": limite, "historico_pay": hist,
                     "ultimo_pagamento": pagto, "risco": round(prob, 4),
                     "previsao_inadimplente": bool(inadimplente), "real": None})
    salvar_clientes(clientes)
    print("Cliente salvo.")


# ---------- 2) Listagem ----------
def listar():
    clientes = carregar_clientes()
    if not clientes:
        print("\nNenhum cliente cadastrado.")
        return
    altos = sum(c["previsao_inadimplente"] for c in clientes)
    print(f"\nTotal: {len(clientes)} clientes | alto risco: {altos} | baixo risco: {len(clientes) - altos}")
    mostrar = clientes[-20:]
    if len(clientes) > 20:
        print("(mostrando os 20 últimos; use a opção 5 para exportar todos em CSV)")
    print(f"\n{'Nome':<25}{'Risco':>8}  Previsão")
    for c in mostrar:
        status = "ALTO RISCO" if c["previsao_inadimplente"] else "baixo risco"
        print(f"{c['nome']:<25}{c['risco']:>8.1%}  {status}")


# ---------- 3) Importação do banco completo ----------
def importar_banco(rede, params):
    print("\n=== IMPORTAR BANCO DE DADOS COMPLETO ===")
    caminho = input(f"Arquivo (.xls/.xlsx/.csv) [{ARQUIVO_DADOS}]: ").strip() or ARQUIVO_DADOS
    if not os.path.exists(caminho):
        print(f"Arquivo '{caminho}' não encontrado.")
        return
    try:
        df = ler_planilha(caminho)
    except Exception as e:
        print(f"Não foi possível ler o arquivo: {e}")
        return

    if "ID" not in df.columns:
        print("Aviso: o arquivo não tem coluna ID, então não há como evitar duplicatas "
              "se ele for importado de novo.")
    # Evita duplicar clientes já importados (mesmo ID)
    clientes = carregar_clientes()
    ids_existentes = {c["id"] for c in clientes if c.get("id") is not None}
    if "ID" in df.columns:
        df = df[~df["ID"].isin(ids_existentes)]
    if df.empty:
        print("Todos os clientes desse arquivo já estavam cadastrados.")
        return

    # Previsão de TODOS de uma vez (operações em matriz, sem loop por cliente)
    probs = rede.prever_proba(normalizar(montar_entradas(df), params)).ravel()
    limiar = float(params["limiar"])

    n = len(df)
    ids = df["ID"].tolist() if "ID" in df.columns else [None] * n
    historicos = df[PAY_COLS].astype(int).values.tolist()
    limites = df["LIMIT_BAL"].astype(float).tolist()
    pagamentos = df["PAY_AMT1"].astype(float).tolist()
    reais = df[COLUNA_ALVO].astype(int).tolist() if COLUNA_ALVO in df.columns else [None] * n

    for i in range(n):
        clientes.append({
            "id": ids[i],
            "nome": f"Cliente {ids[i]}" if ids[i] is not None else f"Cliente {len(clientes) + 1}",
            "limite": limites[i],
            "historico_pay": historicos[i],
            "ultimo_pagamento": pagamentos[i],
            "risco": round(float(probs[i]), 4),
            "previsao_inadimplente": bool(probs[i] >= limiar),
            "real": reais[i],
        })
    salvar_clientes(clientes)
    print(f"{n} clientes importados e analisados. Total cadastrado: {len(clientes)}.")
    relatorio(params)


# ---------- 4) Relatório de acurácia ----------
def relatorio(params):
    clientes = [c for c in carregar_clientes() if c.get("real") is not None]
    if not clientes:
        print("\nNenhum cliente com resultado real conhecido (importe o banco completo - opção 3).")
        return
    y = np.array([[c["real"]] for c in clientes], dtype=float)
    probs = np.array([[c["risco"]] for c in clientes])
    limiar = float(params["limiar"])
    m = metricas(y, (probs >= limiar).astype(int))

    print(f"\n=== RELATÓRIO: {len(clientes)} clientes com resultado real ===")
    print(f"Baseline ('ninguém dá calote'): {1 - y.mean():.1%} | limiar de decisão: {limiar:.2f}")
    print(f"Acurácia {m['acuracia']:.1%} | Precisão {m['precisao']:.1%} | "
          f"Recall {m['recall']:.1%} | F1 {m['f1']:.3f} | AUC-ROC {auc_roc(probs, y):.3f}\n")
    print("Matriz de confusão:")
    imprimir_matriz(m)
    print("\nObs.: parte desses clientes foi usada no treino, então o número é otimista.")


# ---------- 5) Exportação ----------
def exportar():
    clientes = carregar_clientes()
    if not clientes:
        print("\nNenhum cliente para exportar.")
        return
    with open(ARQUIVO_RESULTADOS, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "nome", "limite", "ultimo_pagamento", "risco",
                    "previsao", "real", "acertou"])
        for c in clientes:
            prev = int(c["previsao_inadimplente"])
            real = c.get("real")
            w.writerow([c.get("id"), c["nome"], c["limite"], c["ultimo_pagamento"],
                        f"{c['risco']:.4f}".replace(".", ","), prev,
                        "" if real is None else real,
                        "" if real is None else int(prev == real)])
    print(f"\n{len(clientes)} clientes exportados para '{ARQUIVO_RESULTADOS}'.")


def main():
    # Se o modelo ainda não existe, treina e salva (gera o modelo_mlp.npz)
    if not os.path.exists(ARQUIVO_MODELO):
        if not os.path.exists(ARQUIVO_DADOS):
            print(f"Modelo não encontrado e '{ARQUIVO_DADOS}' também não está na pasta.")
            print("Coloque o banco de dados na pasta para treinar o modelo.")
            return
        print(f"Modelo '{ARQUIVO_MODELO}' não encontrado. Treinando um novo...\n")
        treinar_e_salvar()
        print("\n" + "=" * 60)
    rede, params = MLP.carregar(ARQUIVO_MODELO)

    while True:
        print("\n===== ANÁLISE DE CRÉDITO (MLP) =====")
        print("1) Cadastrar cliente e analisar risco")
        print("2) Listar clientes cadastrados")
        print("3) Importar banco de dados completo")
        print("4) Relatório de acurácia")
        print("5) Exportar resultados (CSV)")
        print("0) Sair")
        op = input("Opção: ").strip()
        if op == "1":
            cadastrar(rede, params)
        elif op == "2":
            listar()
        elif op == "3":
            importar_banco(rede, params)
        elif op == "4":
            relatorio(params)
        elif op == "5":
            exportar()
        elif op == "0":
            break
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    main()
