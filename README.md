# Classificação de Variedades de Trigo

## Dataset

Dados do OpenML sobre propriedades de sementes de 3 variedades de trigo.

## Etapas:

1. Verificação de dados ausentes;
2. Normalização para intervalo [0, 1];
3. Separação de treino (200 instâncias) e teste (10 instâncias);
4. Treinamento com:
   - Árvore de Decisão
   - Naive Bayes
   - SVM
5. Classificação e avaliação da acurácia sobre dados de teste.

Resultado final: acurácia por modelo e visualização do desempenho.

Bibliotecas: `pandas`, `sklearn`, `numpy`, `matplotlib`
