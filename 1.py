from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import MinMaxScaler # Normalização de atributos 
from sklearn.model_selection import train_test_split

dados = pd.read_csv('./phpPrh7lv.csv')

# removendo valores nulos
dados = dados.dropna()

#  Realize a normalização de atributos para o mesmo intervalo [0.0, 1.0]
scaler = MinMaxScaler() #MinMaxScaler: transforma os dados para que fiquem no intervalo [0, 1]
x = pd.DataFrame(dados.iloc[:, :-1]) #todas as linhas e todas as colunas exceto a última
dados_normalizados = pd.DataFrame(scaler.fit_transform(x), columns=x.columns) #columns=x.columns para manter os nomes das colunas

print("DADOS NORMALIZADOS")
print(dados_normalizados)

#3. Separe 200 instâncias aleatoriamente como os dados de treino, e últimas 10 como dados de teste.
X_train, X_test, y_train, y_test = train_test_split(dados_normalizados, dados['Class'], test_size=10, random_state=42)
#X se refere as linhas, y as colunas.
#dados normalizados, dados['Class']: dados de treino, test_size=10 são os dados de teste, random_state lá em baixo

print(f'VARIÁVEIS DE TESTE:')
print(f'',X_test,'\n',y_test,'\n')
# 4. Usando o conjunto de dados de treino, realize aprendizagem para três tipos de classificação: o
# método de árvore de decisão, o método bayesiano, e o método de vetores SVM.

print("\nAPRENDIZAGEM")
# Árvore de decisão 
dtc = DecisionTreeClassifier(random_state=0) #random_state lá em baixo
dtc.fit(X_train, y_train) #.fit() é usado para treinar o modelo
print("Árvore de decisão: ", dtc.score(X_test, y_test))

# Método bayesiano
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("Método bayesiano: ", gnb.score(X_test, y_test))

# Método de vetores suport vector machine (SVM)
svm = SVC()
svm.fit(X_train, y_train)
print("Método de vetores SVM: ", svm.score(X_test, y_test))

#5. Usando os três métodos, realize classificação de conjunto de 10 instâncias de teste.
print("\nCLASSIFICAÇÃO 10 INSTÂNCIAS TESTE")
print("Árvore de decisão: ", dtc.predict(X_test))
print("Método bayesiano: ", gnb.predict(X_test))
print("Método de vetores SVM: ", svm.predict(X_test))

#6. Verifique a acurácia de classificação para os três métodos usando as classes fornecidas junto com os dados medida como o número de instâncias 
# com classes corretas dividido por o número de instâncias de teste.

# Acurácia de classificação para os três métodos
print("\nPRECISÃO DA CLASSIFICAÇÃO")
print("Árvore de decisão:   ", dtc.score(X_test, y_test))
print("Método bayesiano:       ", gnb.score(X_test, y_test))
print("Método de vetores SVM: ", svm.score(X_test, y_test))



# O parâmetro random_state é usado em várias funções em bibliotecas como scikit-learn para controlar a aleatoriedade durante a divisão de dados ou durante 
# a inicialização de modelos que utilizam alguma forma de aleatoriedade internamente.

# Quando você define random_state para um número específico, como 0 ou 42, isso garante que os resultados sejam reproduzíveis. Ou seja, se você executar 
# o mesmo código várias vezes com o mesmo valor de random_state, você obterá os mesmos resultados. Isso é útil para garantir consistência em experimentos ou em situações em que a aleatoriedade pode influenciar os resultados.

# No código que você forneceu, random_state=42 foi usado para garantir que a divisão dos dados em conjuntos de treinamento e teste seja sempre a mesma, 
# independentemente de quantas vezes o código seja executado. O valor 42 é apenas um valor arbitrário escolhido pelo programador.

# Da mesma forma, random_state=0 foi usado ao criar o classificador de árvore de decisão (DecisionTreeClassifier). Isso garante que a inicialização interna 
# do classificador também seja consistente, tornando os resultados reproduzíveis.

# Escolher 0 e 42 não é necessariamente importante em si, mas é comum escolher números que sejam fáceis de lembrar ou que sejam significativos para o 
# contexto do problema. O importante é usar o mesmo valor de random_state sempre que você precisar de resultados reproduzíveis.