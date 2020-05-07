#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm


# In[2]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


athletes = pd.read_csv("athletes.csv")


# In[4]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
athletes.head()


# In[6]:


athletes.describe()


# In[7]:


athletes.info()


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[8]:


height_sample = get_sample(athletes, 'height', n=3000)


# In[9]:


def q1():
    alpha = 0.05
    return bool(sct.shapiro(height_sample)[1] > alpha)

q1()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# In[10]:


sns.distplot(height_sample, bins=25)
plt.show()


# Os resultados não me parecem compatíveis. A distribução assemelha-se bastante à uma distribuição normal. Inclusive, observe o resultado para uma amostra menor:

# In[11]:


sct.shapiro(get_sample(athletes, 'height', n=300))


# Seu p-value é 0.144, o que não nos permite rejeitar a hipótese nula.

# In[12]:


sm.qqplot(sct.zscore(height_sample), line='45')
plt.show()


# O gráfico Q-Q também aponta que os dados sigam a distribuição normal.

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[13]:


def q2():
    alpha = 0.05
    return bool(sct.jarque_bera(height_sample)[1] > alpha)

q2()


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# In[14]:


sct.jarque_bera(height_sample)


# Ainda se rejeita a hipótese nula (os dados não são normais).

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[15]:


weight_sample = get_sample(athletes, 'weight', n=3000)


# In[16]:


def q3():
    alpha = 0.05
    return bool(sct.normaltest(weight_sample)[1] > alpha)
    
q3()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[17]:


f, (ax1, ax2) = plt.subplots(1,2)
sns.distplot(weight_sample, bins=25, ax=ax1)
sns.boxplot(weight_sample, ax=ax2)
plt.show()


# Na coluna 'weight', há um forte skewness positivo.

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[18]:


def q4():
    alpha = 0.05
    return bool(sct.normaltest(np.log(weight_sample))[1] > alpha)

q4()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# In[19]:


f, (ax1, ax2) = plt.subplots(1,2)
sns.distplot(np.log(weight_sample), bins=25, ax=ax1)
sns.boxplot(np.log(weight_sample), ax=ax2)
plt.show()


# Aplicando uma transformação logarítmica, os maiores valores são mais reduzidos do que os valores menores. Esta transformação pode ajudar a reduzir a assimetria positiva em grandezas positivas maiores que 1.
# Apesar disso, ainda há muitos outliers.

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# In[20]:


alpha = 0.05


# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[21]:


bra = athletes[athletes['nationality'] == 'BRA']['height'].dropna()
usa = athletes[athletes['nationality'] == 'USA']['height'].dropna()
can = athletes[athletes['nationality'] == 'CAN']['height'].dropna()


# In[22]:


bra.size, usa.size, can.size


# Como as amostras têm tamanhos diferentes, executamos os testes com a opção `equal_var=False`

# In[23]:


def q5():
    return bool(sct.ttest_ind(bra, usa, equal_var=False)[1] > alpha)

q5()


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[24]:


def q6():
    return bool(sct.ttest_ind(bra, can, equal_var=False)[1] > alpha)

q6()


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[25]:


def q7():
    return float(round(sct.ttest_ind(usa, can, equal_var=False)[1], 8))

q7()


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?
