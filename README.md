# Previsão de Inadimplência de Cartão de Crédito com Rede Neural MLP

Projeto da disciplina de **Inteligência Artificial (UFRPE)**. Uma rede neural **MLP (Multilayer Perceptron)**, implementada **do zero apenas com NumPy**, estima se um cliente de cartão de crédito **deixará de pagar a próxima fatura**, a partir do perfil financeiro e do histórico de pagamentos.

Ao cadastrar um cliente, a aplicação devolve uma **pontuação de risco** e uma recomendação (aprovar ou negar crédito). Também é possível **importar o banco de dados completo** (30.000 clientes), analisá-los de uma só vez e gerar um relatório de acurácia.

---

## Sumário

1. [Visão geral](#visão-geral)
2. [Estrutura do projeto](#estrutura-do-projeto)
3. [Instalação e uso](#instalação-e-uso)
4. [Base de dados](#base-de-dados)
5. [Entradas e saída da rede](#entradas-e-saída-da-rede)
6. [Arquitetura e aprendizado](#arquitetura-e-aprendizado)
7. [Protocolo de treinamento e teste](#protocolo-de-treinamento-e-teste)
8. [Limiar de decisão](#limiar-de-decisão)
9. [Resultados](#resultados)
10. [Validações](#validações)
11. [Experimentos: mais entradas e mais neurônios](#experimentos-mais-entradas-e-mais-neurônios)
12. [Como a aplicação funciona](#como-a-aplicação-funciona)
13. [Solução de problemas](#solução-de-problemas)
14. [Limitações](#limitações)
15. [Referência](#referência)

---

## Visão geral

| Item | Especificação |
|---|---|
| Tipo de problema | Classificação binária (vai ou não dar calote) |
| Modelo | MLP com 1 camada oculta, implementada em NumPy |
| Entradas | 4 |
| Camada oculta | 8 neurônios, ativação **ReLU** |
| Saída | 1 neurônio, ativação **sigmoide** (valor entre 0 e 1) |
| Função de erro | **MSE** (Erro Quadrático Médio) |
| Aprendizado | **Backpropagation** + gradiente descendente em mini-batches |
| Divisão dos dados | **Estratificada**: 70% de cada classe para treino e 30% de cada classe para teste |
| Desbalanceamento | **Oversampling** da classe minoritária (inadimplentes), somente no treino |
| Avaliação | **Matriz de confusão** (VP, FP, VN, FN), acurácia, precisão, recall, F1 e AUC |
| Dados | 30.000 clientes (UCI Machine Learning Repository) |

### Fluxo do projeto

```
Banco (30.000 clientes)
   └─► Divisão estratificada ──► Treino (70% de cada classe) ──► Oversampling da classe 1 ──► Treino da MLP
                             └─► Teste  (30% de cada classe) ─────────────────────────────► Teste + matriz de confusão
                                                                                              │
                                    modelo_mlp.npz (pesos + normalização + limiar) ◄──────────┘
                                                │
                                                └─► app.py (cadastro, importação do banco, relatório e exportação)
```

---

## Estrutura do projeto

```
credito_mlp/
├── mlp.py                 # A rede neural (forward, backpropagation, treino, salvar/carregar)
├── treinamento.py         # Dados, divisão estratificada, oversampling, treino, teste e salvamento do modelo
├── app.py                 # Aplicação (menu): cadastro, importação do banco, relatório e exportação
├── avaliar_modelo.py      # Confere o modelo salvo (sem retreinar): banco completo, treino e teste
├── validacoes.py          # Bateria de 19 testes que comprova o funcionamento do projeto
├── colab_credito_mlp.ipynb # Notebook para rodar o projeto no Google Colab (aula e apresentação)
├── modelo_mlp.npz         # Modelo treinado (pesos, normalização e limiar)
├── curva_aprendizado.png  # Gráfico do MSE por época
├── requirements.txt       # Dependências
├── .gitignore             # Arquivos que não devem ir para o GitHub (dados de clientes, .venv etc.)
└── default_of_credit_card_clients.xls   # Banco de dados (baixar do UCI, ver abaixo)
```

### Qual arquivo usar para quê

| Arquivo | Para que serve | Quando rodar |
|---|---|---|
| `treinamento.py` | **Cria** o modelo: divide os dados, faz o oversampling, treina, **testa nos 30% restantes**, imprime a matriz de confusão e salva o modelo | Quando quiser treinar (ou retreinar) a rede |
| `app.py` | Usa o modelo salvo para analisar clientes (um a um ou o banco inteiro) | Para usar a aplicação |
| `avaliar_modelo.py` | **Confere** o modelo **já salvo**, sem retreinar: compara o banco completo, o treino (70%) e o teste (30%), com a matriz de confusão | Para verificar o modelo, por exemplo mostrar que treino e teste dão o mesmo resultado |
| `validacoes.py` | Prova que o código está correto (gradiente, divisão, oversampling, importação etc.) | Antes de entregar ou ao alterar o código |
| `colab_credito_mlp.ipynb` | Notebook que monta o projeto no Google Colab, treina, mostra a matriz de confusão, roda as validações e abre a aplicação | Para apresentar ou testar em sala de aula, sem instalar nada |
| `mlp.py` | A rede neural em si, usada pelos demais | Não é executado diretamente |

O `avaliar_modelo.py` **não substitui** o `treinamento.py` e vice-versa: o primeiro só mede o que o segundo produziu. O número que mede a capacidade de generalização do modelo é o do **teste**, que o próprio `treinamento.py` já imprime; o do banco completo é otimista, porque inclui os clientes usados no treino.

---

## Instalação e uso

**Requisitos:** Python 3.9 ou superior.

```bash
pip install -r requirements.txt
```

`numpy` e `pandas` são o núcleo do projeto; `python-calamine` lê `.xls` e `.xlsx`; e `matplotlib` só gera o gráfico da curva de aprendizado. Se o `python-calamine` não estiver instalado, o projeto usa `xlrd` (para `.xls`) ou `openpyxl` (para `.xlsx`), que precisam ser instalados à parte e, nas versões novas do pandas, geram um aviso de descontinuação (inofensivo).

Baixe o banco de dados (veja [Base de dados](#base-de-dados)) e coloque o arquivo `default_of_credit_card_clients.xls` na mesma pasta dos scripts.

```bash
python treinamento.py     # treina a rede, testa nos 30% restantes e gera modelo_mlp.npz e curva_aprendizado.png
python app.py             # abre a aplicação (se o modelo_mlp.npz não existir, treina e salva um novo)
python avaliar_modelo.py  # confere o modelo salvo em todos os 30.000 clientes
                          # (se o modelo_mlp.npz não existir, treina e salva um novo antes)
python validacoes.py      # roda as 19 validações do projeto
```

O repositório já inclui um `modelo_mlp.npz` treinado, então é possível rodar `app.py` direto. Se o arquivo for removido, tanto o `app.py` quanto o `avaliar_modelo.py` treinam e salvam um novo automaticamente (é necessário que o `.xls` esteja na pasta). O modelo gerado é sempre o mesmo, pois todas as sementes aleatórias são fixas.

### Menu da aplicação

| Opção | Função |
|---|---|
| 1 | Cadastrar um cliente e analisar o risco |
| 2 | Listar clientes cadastrados (totais de alto e baixo risco e os 20 últimos) |
| 3 | Importar o banco de dados completo (`.xls`, `.xlsx` ou `.csv`) e analisar todos de uma vez |
| 4 | Relatório de acurácia dos clientes que têm resultado real conhecido |
| 5 | Exportar os resultados para `resultados_clientes.csv` |
| 0 | Sair |

### Executando no Google Colab

O notebook `colab_credito_mlp.ipynb` roda o projeto inteiro no navegador, sem instalar nada. O Colab já traz NumPy, pandas e matplotlib; uma CPU comum é suficiente (não precisa de GPU).

1. Acesse [colab.research.google.com](https://colab.research.google.com) e entre com a conta Google.
2. Vá em **Arquivo → Fazer upload de notebook** e escolha `colab_credito_mlp.ipynb`.
3. Execute a primeira célula de código. Quando a janela de envio abrir, selecione **`credito_mlp.zip`** (o projeto compactado) e **`default_of_credit_card_clients.xls`** (o banco), os dois de uma vez.
4. Execute as demais células em ordem (**Ctrl + Enter**, ou *Ambiente de execução → Executar tudo*). O notebook treina, mostra a curva de aprendizado e a matriz de confusão, roda as validações, testa um cliente e abre o menu da aplicação.

Observações:

- **Os arquivos enviados ao Colab somem quando a sessão termina.** Na próxima aula, basta rodar a primeira célula de novo. Para evitar o reenvio do projeto, use o GitHub (abaixo).
- **Pelo GitHub:** com o projeto em um repositório, cole a URL (terminada em `.git`) na variável `URL_GITHUB` da primeira célula e o código é baixado de lá. O banco `.xls` ainda precisa ser enviado se não estiver no repositório. Também dá para abrir o notebook direto do repositório pelo endereço `https://colab.research.google.com/github/SEU-USUARIO/SEU-REPOSITORIO/blob/main/colab_credito_mlp.ipynb` (troque pelo seu usuário e repositório; o ramo principal precisa se chamar `main`).
- **Menu da aplicação:** a última célula abre o menu e espera as opções digitadas no campo que aparece logo abaixo dela.

---

## Base de dados

**Default of Credit Card Clients**, do UCI Machine Learning Repository, com dados de clientes de cartão de crédito de Taiwan (2005).

- **30.000 clientes**, 23 variáveis explicativas e 1 variável-alvo (`default payment next month`: 1 = não pagou, 0 = pagou).
- **22,1% dos clientes são inadimplentes** (6.636 da classe 1 contra 23.364 da classe 0): a base é **desbalanceada**. Um modelo que sempre responde "vai pagar" acerta 77,9% sem detectar nenhum calote. Por isso a acurácia sozinha não basta, e o projeto trata o desbalanceamento com oversampling e também reporta precisão, recall, F1 e AUC.
- Variáveis originais: limite de crédito, sexo, escolaridade, estado civil, idade, situação de pagamento dos últimos 6 meses (`PAY_0`, `PAY_2` a `PAY_6`), valor das faturas (`BILL_AMT1` a `BILL_AMT6`) e valor dos pagamentos (`PAY_AMT1` a `PAY_AMT6`).

Codificação das colunas `PAY_*`:

| Valor | Significado |
|---|---|
| -2 | Não usou o cartão no mês |
| -1 | Pagou a fatura inteira |
| 0 | Pagou apenas o mínimo (crédito rotativo) |
| 1 a 9 | Meses de atraso |

> Os arquivos `clientes.json` e `resultados_clientes.csv` são gerados pela aplicação e contêm os dados dos clientes. Recomenda-se colocá-los no `.gitignore`.

---

## Entradas e saída da rede

### As 4 entradas

A escolha foi feita analisando cada variável isoladamente (AUC e correlação com o calote) e comparando conjuntos de 4 entradas treinando a própria MLP.

| Entrada | Como é calculada | Por que foi escolhida |
|---|---|---|
| `PAY_0` | Situação do pagamento no mês mais recente | Variável original mais forte (AUC isolado ≈ 0,69) |
| `PAY_MAX` | Maior valor entre os 6 meses de situação de pagamento (pior atraso) | Melhor variável de todas (AUC isolado ≈ 0,71); resume as 6 colunas de histórico em uma |
| `LOG_LIMITE` | `log(1 + limite de crédito)` | Representa o perfil financeiro e é pouco correlacionada com o histórico |
| `LOG_PAGTO` | `log(1 + valor pago na última fatura)` | Traz o comportamento de pagamento, que os status de atraso não capturam |

O logaritmo é usado porque valores em dinheiro são muito assimétricos (poucos clientes com valores enormes).

**Variáveis descartadas:** idade, sexo, escolaridade e estado civil têm AUC próximo de 0,50, ou seja, quase não ajudam a prever o calote. Trocar `LOG_PAGTO` pela idade piorou o resultado. Além disso, usar sexo ou estado civil em decisões de crédito levanta questões éticas de discriminação.

### A saída

Um único neurônio com **sigmoide**, que produz uma **pontuação de risco** entre 0 e 1. Quanto maior a pontuação, maior o risco; a partir de 0,5 o cliente é classificado como **alto risco**.

**Atenção: a pontuação não é a probabilidade real de calote.** Como a rede é treinada com as duas classes em quantidades iguais (oversampling), ela "enxerga" metade de inadimplentes, quando na realidade são 22%. A pontuação média da rede é 43%, contra 22% de calotes reais. O que se mantém é a **ordem**: pontuação maior significa risco real maior. Medindo nos 30.000 clientes:

| Pontuação da rede | Clientes | Taxa real de calote |
|---|---|---|
| 0,0 a 0,3 | 11.148 | 8,0% |
| 0,3 a 0,4 | 4.772 | 14,2% |
| 0,4 a 0,5 | 4.796 | 18,2% |
| 0,5 a 0,6 | 3.066 | 26,0% |
| 0,6 a 0,7 | 2.067 | 36,1% |
| 0,7 a 1,0 | 4.151 | 63,9% |

---

## Arquitetura e aprendizado

```
 4 entradas ──► 8 neurônios ocultos (ReLU) ──► 1 neurônio de saída (sigmoide) ──► pontuação de risco (0 a 1)
```

**Parâmetros da rede:** `W1` (4×8), `b1` (1×8), `W2` (8×1) e `b2` (1×1), ou seja, 49 parâmetros. A inicialização é aleatória: **He** na camada ReLU (desvio √(2/entradas)) e **Xavier** na saída (desvio √(1/ocultos)).

### Forward (passo para frente)

```
Z1 = X · W1 + b1        A1 = ReLU(Z1) = max(0, Z1)
Z2 = A1 · W2 + b2       ŷ  = sigmoide(Z2) = 1 / (1 + e^(-Z2))
```

### Erro (MSE)

```
MSE = média( (ŷ − y)² )
```

### Backpropagation

Aplicação da regra da cadeia, da saída para a entrada (implementada em `MLP.backward`):

```
1. dMSE/dŷ  = 2 (ŷ − y) / n               derivada do erro
2. dZ2      = dMSE/dŷ · ŷ (1 − ŷ)         passa pela sigmoide
3. dW2      = A1ᵀ · dZ2     db2 = Σ dZ2   gradientes da camada de saída
4. dA1      = dZ2 · W2ᵀ                    propaga o erro para a camada oculta
5. dZ1      = dA1 · ReLU'(Z1)              ReLU' = 1 se Z1 > 0, senão 0
6. dW1      = Xᵀ · dZ1      db1 = Σ dZ1   gradientes da camada oculta
7. peso     = peso − taxa_de_aprendizado · gradiente   (gradiente descendente)
```

A implementação do backpropagation foi comprovada por **verificação numérica de gradiente** (veja [Validações](#validações)).

---

## Protocolo de treinamento e teste

O treino segue o roteiro: **dividir de forma estratificada, balancear só o treino por oversampling, treinar, e testar nos 30% restantes**.

| Passo | O que é feito | Função em `treinamento.py` |
|---|---|---|
| 1 | **Divisão estratificada:** 70% dos exemplos da classe 0 + 70% dos exemplos da classe 1 vão para o treino; os 30% restantes de cada classe vão para o teste | `dividir_estratificado` |
| 2 | **Oversampling** da classe 1 no treino: sorteio com reposição de inadimplentes até igualar o número de exemplos da classe 0 | `balancear_oversampling` |
| 3 | **Treino** da MLP com o conjunto balanceado | `MLP.treinar` |
| 4 | **Teste** nos 30% restantes (que nunca passaram por oversampling) e cálculo da **matriz de confusão** (VP, FP, VN, FN) | `treinar_e_salvar`, `metricas` |

Quantidades:

| Conjunto | Total | Classe 0 (paga) | Classe 1 (calote) | % de calotes |
|---|---|---|---|---|
| Banco original | 30.000 | 23.364 | 6.636 | 22,1% |
| Treino (70%) | 21.000 | 16.355 | 4.645 | 22,1% |
| **Treino balanceado** | **32.710** | **16.355** | **16.355** | **50,0%** |
| Teste (30%) | 9.000 | 7.009 | 1.991 | 22,1% |

A classe 1 do treino tem 4.645 clientes; o oversampling acrescenta 11.710 cópias sorteadas desses mesmos clientes. Nenhum cliente novo é inventado.

**Por que o oversampling vem depois da divisão.** Se fosse feito antes, cópias do mesmo cliente cairiam no treino e no teste, e o modelo seria testado em exemplos que já viu (vazamento de dados). Por isso o teste mantém a proporção real de 22% de inadimplentes, e a avaliação reflete o que aconteceria com clientes novos.

### Hiperparâmetros

| Item | Valor |
|---|---|
| Normalização | Média 0 e desvio 1, calculados **somente no treino original** (antes do oversampling) |
| Taxa de aprendizado | 0,05 |
| Épocas | 100 |
| Tamanho do lote (mini-batch) | 64, com embaralhamento a cada época |
| Semente aleatória | 42 (divisão, oversampling, pesos iniciais e embaralhamento) |

A normalização é necessária porque as entradas têm escalas muito diferentes (por exemplo, status de atraso de −2 a 9 contra o log do limite). A média e o desvio ficam salvos no `modelo_mlp.npz` e são reaplicados no cadastro.

**Sobre a taxa de aprendizado.** Com o treino balanceado, a taxa de 0,5 (usada antes do oversampling) deixou a curva instável: o MSE oscilava entre 0,17 e 0,22 de uma época para outra, e o modelo final dependia da sorte da última época. A taxa foi reduzida para 0,05, escolhida pela **estabilidade da curva** de aprendizado, e não pelo melhor resultado no teste: entre 0,05 e 0,02 os resultados são praticamente iguais.

![Curva de aprendizado](curva_aprendizado.png)

A curva mostra o MSE do treino balanceado (o que a rede minimiza, que cai de forma contínua de 0,1953 na época 10 para 0,1934 na época 100) e o MSE em treino e teste na distribuição original. As curvas de treino original e de teste ficam praticamente sobrepostas (0,1842 e 0,1836 ao final) e oscilam levemente, porque o MSE na distribuição original é sensível a pequenos deslocamentos do ponto de corte da rede.

---

## Limiar de decisão

O cliente é classificado como **alto risco** quando a pontuação da rede é **maior ou igual a 0,5**.

Como o treino é balanceado (50% de cada classe), 0,5 é o ponto de corte natural: a rede aprende a separar os dois grupos como se fossem igualmente frequentes. Antes do oversampling, o limiar precisava ser ajustado à mão (chegou a 0,25) para a rede acusar algum risco numa base com 22% de calotes. Agora o oversampling cumpre esse papel, e **nenhum limiar é ajustado** com dados de teste.

Apenas para ilustrar o compromisso entre acertar calotes e evitar alarmes falsos, este é o resultado no teste se o limiar fosse outro (informativo, não foi usado para escolher nada):

| Limiar | Marcados como alto risco | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|---|
| 0,3 | 62,7% | 53,4% | 30,5% | 86,4% | 0,451 |
| 0,4 | 46,8% | 64,7% | 35,9% | 75,8% | 0,487 |
| **0,5 (usado)** | **30,8%** | **74,9%** | **45,2%** | **62,9%** | **0,526** |
| 0,6 | 20,9% | 80,0% | 55,0% | 51,9% | 0,534 |
| 0,7 | 14,2% | 82,0% | 64,7% | 41,4% | 0,505 |

Subir o limiar reduz os alarmes falsos, mas deixa passar mais calotes; descer faz o contrário. A escolha ideal depende de qual erro custa mais para a instituição: conceder crédito a quem não paga ou negar crédito a quem pagaria.

"Alto risco" significa que o cliente pertence a um grupo com taxa de calote **bem acima da média** (26% a 64% de calotes nas faixas acima de 0,5, contra 22% no banco todo), e não que ele certamente deixará de pagar. Clientes com pontuação muito próxima de 0,5 recebem rótulos opostos apesar de risco praticamente igual.

---

## Resultados

### Conjunto de teste (9.000 clientes, os 30% restantes, nunca vistos no treino)

| Métrica | Valor |
|---|---|
| Acurácia | 74,9% |
| Precisão | 45,2% |
| Recall | 62,9% |
| F1 | 0,526 |
| AUC-ROC | 0,771 |

**Matriz de confusão do teste:**

| | Previsto: paga (0) | Previsto: calote (1) |
|---|---|---|
| **Real: paga (0)** | VN = 5.491 | FP = 1.518 |
| **Real: calote (1)** | FN = 738 | VP = 1.253 |

Dos 1.991 clientes que realmente deram calote, o modelo detectou 1.253 (62,9%). Dos 7.009 bons pagadores, acertou 5.491 (78,3%) e acusou risco em 1.518.

### Banco completo (30.000 clientes)

| Métrica | Todos os clientes | Só treino (21.000) | Só teste (9.000) |
|---|---|---|---|
| Acurácia | 74,9% | 74,9% | 74,9% |
| Precisão | 45,2% | 45,2% | 45,2% |
| Recall | 63,2% | 63,3% | 62,9% |
| F1 | 0,527 | 0,527 | 0,526 |
| AUC-ROC | 0,770 | 0,770 | 0,771 |

Matriz de confusão no banco completo: VN = 18.274, FP = 5.090, FN = 2.442, VP = 4.194.

### Como interpretar

- **Acurácia e baseline.** O chute "ninguém dá calote" acerta 77,9%, mas detecta 0% dos calotes. O modelo fica abaixo dessa acurácia porque aceita acusar risco em clientes que pagariam, em troca de detectar 62,9% dos calotes. Por isso **recall, F1 e AUC** são as métricas mais informativas aqui, junto com a matriz de confusão.
- **Sem overfitting.** Treino e teste têm as mesmas métricas (74,9% de acurácia, AUC 0,770 e 0,771).
- **Efeito do oversampling.** Com o protocolo anterior (divisão aleatória, sem balanceamento e limiar de 0,25 ajustado à mão), o resultado foi parecido (76,1% de acurácia, 61,8% de recall, AUC 0,767, num teste diferente de 15%). O ganho do oversampling não está em números muito maiores, e sim em **dispensar o ajuste manual do limiar** e deixar o treino equilibrado entre as classes.
- **Teto do problema.** Um AUC em torno de 0,77 a 0,79 é típico para esse banco de dados, mesmo com modelos mais complexos.

---

## Validações

O projeto passou por três tipos de validação: **estatística** (o modelo generaliza?), **numérica** (o código de aprendizado está correto?) e **funcional** (a aplicação se comporta bem?). As validações numéricas e funcionais estão automatizadas em `validacoes.py`.

### 1. Validação estatística do modelo

| Verificação | Como foi feita | Resultado |
|---|---|---|
| Generalização (overfitting) | Métricas no treino original e no teste | Acurácia 74,9% nos dois; AUC 0,770 e 0,771; MSE 0,1842 e 0,1836 |
| Curva de aprendizado | MSE por época em treino balanceado, treino original e teste (`curva_aprendizado.png`) | O MSE do treino balanceado cai de forma contínua; treino original e teste ficam sobrepostos |
| Dados nunca vistos | Avaliação em 9.000 clientes separados (30%), sem oversampling | AUC 0,771, recall 62,9% |
| Estratificação | Proporção de inadimplentes no banco, no treino e no teste | 22,12% nos três |
| Balanceamento | Contagem das classes no treino antes e depois do oversampling | 16.355 e 4.645 antes; 16.355 e 16.355 depois |
| Comparação com um chute | Baseline "ninguém dá calote" | 77,9% de acurácia, mas 0% de recall; o modelo detecta 62,9% dos calotes |
| Limiar | Fixo em 0,5, sem ajuste com dados de teste | Ver [Limiar de decisão](#limiar-de-decisão) |
| Pontuação da rede | Taxa real de calote por faixa de pontuação | Cresce de 8,0% a 63,9%, acompanhando a pontuação (tabela em [A saída](#a-saída)) |
| Escolha das entradas | AUC isolado de cada variável e comparação de conjuntos treinando a MLP | `PAY_0`, `PAY_MAX`, limite e último pagamento; idade, sexo, escolaridade e estado civil ≈ 0,50 |
| Tamanho da rede | Grade de 5 conjuntos de entradas × 5 tamanhos de camada oculta | Ver [Experimentos](#experimentos-mais-entradas-e-mais-neurônios) |

### 2. Validações automatizadas (`python validacoes.py`)

Resultado da última execução: **19 de 19 validações passaram** (cerca de 30 segundos).

**A) A rede neural**

| Validação | O que comprova | Resultado |
|---|---|---|
| Gradiente numérico | O backpropagation calcula os mesmos gradientes que a derivada numérica `(f(w+ε) − f(w−ε)) / 2ε` para `W1`, `b1`, `W2` e `b2` | Erro relativo máximo de 1,1 × 10⁻⁹ |
| Saída entre 0 e 1 | A sigmoide mantém a saída válida, inclusive com entradas extremas (escala 50×) | Formato (n, 1), valores finitos entre 0 e 1 |
| Problema XOR | A rede 2-8-1 com ReLU, sigmoide e MSE aprende um problema não linear que um neurônio sozinho não resolve | MSE de 0,261 para 0,0001; 4 de 4 acertos |
| Métricas | Acurácia, precisão, recall, F1, matriz de confusão e AUC conferidos em casos pequenos com resposta conhecida | Corretas |

**B) Os dados e o protocolo de treino**

| Validação | O que comprova | Resultado |
|---|---|---|
| Integridade do banco | 30.000 linhas, IDs únicos, sem valores nulos nas colunas usadas, alvo só com 0 e 1, pagamentos não negativos | 22,1% de inadimplentes; `PAY` de −2 a 8; limite de 10.000 a 1.000.000 |
| **Divisão estratificada** | 70% da classe 0 e 70% da classe 1 no treino, 30% de cada no teste, sem nenhum cliente repetido entre os dois | Treino 21.000 (16.355 e 4.645); teste 9.000 (7.009 e 1.991) |
| **Mesma proporção** | Treino e teste têm a mesma proporção de inadimplentes do banco | 22,12% nos três |
| **Oversampling iguala as classes** | Treino balanceado tem o mesmo número de exemplos nas duas classes | 16.355 e 16.355 |
| **Oversampling sem vazamento** | Nenhum cliente do treino é perdido, só a classe 1 é duplicada, e nenhum cliente do teste entra no treino | 21.000 clientes distintos, iguais ao treino original |
| Sem vazamento na normalização | A média e o desvio salvos no modelo vêm só do treino original | Coincidem |

**C) O modelo salvo**

| Validação | O que comprova | Resultado |
|---|---|---|
| Arquitetura | O arquivo tem `W1` 4×8, `W2` 8×1 e limiar entre 0 e 1 | Limiar 0,50 |
| **Matriz de confusão** | No teste, VP + FP + VN + FN = 9.000; VP + FN é o total de calotes reais; VN + FP é o total de bons pagadores; e os valores coincidem com as métricas | VP = 1.253, FP = 1.518, VN = 5.491, FN = 738 |
| Reprodutibilidade | Retreinar do zero, numa pasta temporária, gera exatamente os mesmos pesos, normalização e limiar | Idêntico |

**D) A aplicação**

| Validação | O que comprova | Resultado |
|---|---|---|
| Entradas do cadastro | Rejeita texto, valores fora da faixa e decimais onde se exige inteiro; aceita vírgula decimal | Aceitos: 2000, 3 e 1,5 |
| Importação completa | Importa os 30.000 clientes e **não duplica** ao importar o mesmo arquivo de novo | 30.000 na 1ª e na 2ª importação |
| Lote × cadastro | A previsão da importação em lote é igual à do cadastro individual | 300 clientes; maior diferença 5,0 × 10⁻⁵ (arredondamento a 4 casas) |
| Exportação | O CSV tem cabeçalho e uma linha por cliente | 30.001 linhas |
| Formatos de arquivo | A importação lê `.csv`, `.xlsx` e variações do dataset (`PAY_1`, `default.payment.next.month`), aceita arquivo sem `ID` e sem resultado real, e informa qual coluna está ausente | 5 de 5 formatos com entradas idênticas às do `.xls` |
| Arquivos ausentes | Sem modelo e sem banco, o app avisa em vez de quebrar | Mensagem clara |

Os testes que gravam arquivos usam pastas temporárias, então **não alteram** o seu `modelo_mlp.npz` nem o seu `clientes.json`.

### 3. Validações embutidas na aplicação

| Situação | O que o app faz |
|---|---|
| Limite do cartão menor que 1 ou texto no lugar de número | Pede o valor novamente |
| Situação de pagamento fora de −2 a 9 ou não inteira | Pede o valor novamente |
| Valor pago negativo | Pede o valor novamente |
| Importação de arquivo inexistente ou ilegível | Mostra o motivo e volta ao menu |
| Colunas obrigatórias ausentes no arquivo importado | Informa quais colunas faltam |
| Importar o mesmo banco duas vezes | Ignora clientes com `ID` já cadastrado |
| Arquivo importado sem coluna `ID` | Avisa que não há como evitar duplicatas se for importado de novo |
| Modelo ausente | Treina e salva um novo automaticamente |
| Modelo e banco ausentes | Avisa que o `.xls` precisa estar na pasta |
| Relatório sem clientes com resultado real | Orienta a importar o banco completo |

---

## Experimentos: mais entradas e mais neurônios

Variando o número de entradas e de neurônios ocultos, com o mesmo protocolo do projeto (divisão estratificada, oversampling, taxa 0,05, 100 épocas e limiar 0,5), medido no conjunto de teste:

| Entradas | AUC | Acurácia | Recall | F1 |
|---|---|---|---|---|
| 4 (atual) | 0,771 | 74,9% | 62,9% | 0,526 |
| 6 | 0,776 | 75,3% | 62,0% | 0,526 |
| 8 | 0,779 | 76,0% | 62,3% | 0,535 |
| 10 | 0,784 | 75,9% | 64,1% | 0,541 |
| 29 (todas as colunas disponíveis, incluindo as derivadas) | 0,785 | 75,5% | 65,4% | 0,541 |

*(Todas com 8 neurônios ocultos. As entradas extras são, nesta ordem: `PAY_2` e a fatura dividida pelo limite; atraso médio e número de meses em atraso; pagamento dividido pela fatura e pagamento médio.)*

Variando os neurônios ocultos (AUC no teste):

| Entradas | 4 neurônios | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| 4 (atual) | 0,773 | 0,771 | 0,774 | 0,773 | 0,773 |
| 10 | 0,782 | 0,784 | 0,786 | 0,784 | 0,778 |
| 29 | 0,783 | 0,785 | 0,779 | 0,775 | 0,763 |

- **Mais entradas ajudam pouco.** Passar de 4 para 10 entradas sobe o AUC em cerca de 0,013 e o F1 em cerca de 0,015.
- **Mais neurônios não ajudam, e demais atrapalham.** Entre 4 e 16 neurônios o resultado é parecido. Com 64 neurônios o resultado piora, principalmente com muitas entradas (AUC 0,763 com 29 entradas): a rede provavelmente passa a decorar os exemplos duplicados pelo oversampling (overfitting).
- **Conclusão:** a configuração pedida (4 entradas e 8 neurônios) já está perto do teto desses dados. A melhor combinação medida foi 10 entradas com 8 a 16 neurônios (AUC 0,784 a 0,786), um ganho modesto.
- **Ressalva:** como o protocolo tem só treino e teste (sem conjunto de validação separado), essas comparações usaram o teste, então são **exploratórias**. Cada configuração foi treinada uma única vez, e diferenças de 0,005 a 0,01 no AUC podem ser ruído. As 4 entradas do projeto foram definidas antes, na análise de variáveis, e não saíram dessa tabela.

Mudanças fora da especificação, como entropia cruzada no lugar do MSE, otimizador Adam ou parada antecipada, não foram testadas.

---

## Como a aplicação funciona

### Cadastro individual (opção 1)

O usuário informa nome, limite do cartão, a situação de pagamento dos 6 últimos meses e o valor pago na última fatura. A aplicação calcula `PAY_0` e `PAY_MAX` sozinha, aplica o logaritmo e a normalização do treino, passa os valores pela rede e mostra:

- a **pontuação de risco** (saída da rede; veja o aviso em [A saída](#a-saída));
- a **previsão**: alto risco (pontuação ≥ 0,5) ou baixo risco, descrita como "perfil com maior chance de não pagar" ou "de pagar", e não como certeza;
- uma **sugestão**: crédito negado, com limite reduzido ou em análise manual, ou crédito aprovado;
- um lembrete de que a pontuação serve para ordenar clientes por risco e não é a probabilidade real de calote.

### Importação do banco completo (opção 3)

- Lê `.xls`, `.xlsx` ou `.csv` e aceita variações do dataset (`PAY_1` no lugar de `PAY_0`, `default.payment.next.month`).
- Faz a previsão dos 30.000 clientes **de uma só vez**, em operações de matriz, em poucos segundos.
- Ignora clientes cujo `ID` já foi importado, evitando duplicatas. Se o arquivo não tiver coluna `ID`, avisa que isso não é possível.
- Ao final, mostra o relatório de acurácia.

### Relatório e exportação (opções 4 e 5)

O relatório mostra acurácia, precisão, recall, F1, matriz de confusão e AUC dos clientes com resultado real conhecido. A exportação gera `resultados_clientes.csv` (separador `;`) com as colunas:

| Coluna | Conteúdo |
|---|---|
| `id` | Identificador do cliente (vazio nos cadastros manuais) |
| `nome` | Nome ou "Cliente ID" |
| `limite` | Limite do cartão |
| `ultimo_pagamento` | Valor pago na última fatura |
| `risco` | Saída da rede (0 a 1) |
| `previsao` | 1 = alto risco, 0 = baixo risco |
| `real` | Resultado real, quando conhecido |
| `acertou` | 1 se a previsão coincide com o real |

---

## Solução de problemas

| Mensagem ou sintoma | Causa | Solução |
|---|---|---|
| `TypeError: MLP.treinar() got an unexpected keyword argument 'monitorar'` (ou erro parecido de função ou parâmetro que não existe) | Arquivos de **versões diferentes** misturados na mesma pasta, por exemplo um `mlp.py` antigo com um `treinamento.py` novo | Copie **todos** os arquivos `.py` do projeto de uma vez, nunca só alguns. Para conferir, rode `python validacoes.py` |
| `Pandas4Warning: The xlrd engine is deprecated ...` | O pandas novo avisa que o leitor de `.xls` (xlrd) será descontinuado. Não é um erro | `pip install python-calamine`; o projeto passa a usá-lo automaticamente |
| No Colab: `ModuleNotFoundError: No module named 'app'` (ou `O projeto não está preparado nesta sessão`) | A célula **1. Preparar o projeto** não foi executada nesta sessão. Quando a sessão do Colab reinicia ou desconecta, os arquivos enviados são apagados e a pasta volta ao início | Execute a célula 1 de novo e envie o `credito_mlp.zip` e o `.xls`; depois rode as demais em ordem. Não é preciso mexer no `sys.path` |
| `Modelo 'modelo_mlp.npz' não encontrado. Treinando um novo...` | O arquivo do modelo não está na pasta | Nada a fazer: o treino é feito e o modelo é salvo sozinho (o `.xls` precisa estar na pasta) |
| `Banco de dados '...xls' não encontrado na pasta` | O arquivo `default_of_credit_card_clients.xls` não está na mesma pasta dos scripts | Coloque o banco na pasta do projeto |
| `Colunas ausentes no arquivo: [...]` | O arquivo importado não tem as colunas necessárias (`LIMIT_BAL`, `PAY_0` a `PAY_6` e `PAY_AMT1`) | Use o banco do UCI ou um arquivo com essas colunas |

---

## Limitações

- **Sem conjunto de validação.** O protocolo usa só treino (70%) e teste (30%). Os hiperparâmetros (taxa de aprendizado, épocas) foram definidos pela estabilidade da curva de treino, e não por um conjunto de validação separado.
- **Uma única divisão dos dados.** Os resultados vêm de uma divisão com semente fixa. Não foi feita validação cruzada (k-fold), então não há estimativa da variação do resultado entre divisões diferentes.
- **Oversampling por duplicação.** A classe minoritária é completada com cópias de clientes já existentes. Isso equilibra o treino, mas não acrescenta informação nova e pode levar a rede a decorar esses clientes (o que explicaria a piora observada com 64 neurônios, hipótese não testada).
- **Pontuação não é probabilidade.** Por causa do balanceamento, a pontuação da rede superestima a chance real de calote. Use-a para ordenar clientes por risco, não como porcentagem de chance.
- **Entradas fora da faixa vista.** O limite no banco vai de 10.000 a 1.000.000 NT$ (mediana 140.000). Valores muito diferentes, como um limite de 1.000, fazem a rede extrapolar e o resultado perde confiabilidade. O app não bloqueia esses valores.
- **Precisão moderada.** Cerca de metade dos clientes marcados como alto risco paga normalmente (precisão de 45,2% no teste).
- **Dados antigos e de uma região.** O banco é de Taiwan, de 2005; o modelo não deve ser usado para decisões reais de crédito.
- **Ética.** O modelo não usa sexo, idade, escolaridade nem estado civil, o que reduz o risco de discriminação, mas qualquer uso real exigiria uma análise de viés e conformidade legal.
- **Projeto didático.** O objetivo é estudar o funcionamento de uma MLP e do backpropagation, por isso não há bibliotecas de aprendizado de máquina (scikit-learn, TensorFlow etc.) no treino.

---

## Referência

Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473-2480.

Base de dados: UCI Machine Learning Repository, *Default of Credit Card Clients*. Consulte a página do dataset para as condições de uso e licença antes de redistribuir o arquivo.
