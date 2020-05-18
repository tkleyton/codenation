#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
countries.info()


# In[6]:


float_cols = [
    "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]
for col in float_cols:
    countries[col] = countries[col].str.replace(',', '.').astype(float)
    
countries.info()


# In[7]:


strip_cols = [
    "Country", "Region"
]

for col in strip_cols:
    countries[col] = countries[col].transform(str.strip)
countries.head()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[8]:


def q1():
    return countries["Region"].drop_duplicates().sort_values().to_list()


q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[9]:


from sklearn.preprocessing import KBinsDiscretizer


def q2():
    est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    binned = est.fit_transform(countries['Pop_density'].to_numpy().reshape(-1,1))
    return len(binned[binned==9])


q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[10]:


from sklearn.preprocessing import OneHotEncoder


def q3():
    # One-hot encoding cria n atributos sendo n o número de categorias únicas.
    # Porém, há uma variável redundante, logo idealmente cada encoding produz n-1 atributos.
    # Aparentemente o resultado esperado também classifica NaNs como um atributo a parte.
    countries['Climate'].fillna(0, inplace=True)
    ohe = OneHotEncoder(dtype=np.int32)
    encoded = ohe.fit_transform(countries[['Region', 'Climate']])
    return encoded.shape[1]


q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[11]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[12]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def q4():
    numeric_cols = [cname for cname in countries.columns if countries[cname].dtype in ['int64', 'float64']]
    model = Pipeline([
        ("fill", SimpleImputer(strategy='median')),
        ("standardize", StandardScaler())
    ])
    model.fit(countries[numeric_cols])
    df_test_country = pd.DataFrame([dict(zip(new_column_names, test_country))])
    arable_idx = numeric_cols.index('Arable')
    return float(round(model.transform(df_test_country[numeric_cols])[0][arable_idx], 3))


q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[13]:


sns.boxplot(countries['Net_migration'])


# In[14]:


def q5():
    x = countries['Net_migration']
    q1, q3 = x.quantile([.25, .75])
    iqr = q3 - q1
    outliers_abaixo = int(x[x < q1 - 1.5*iqr].shape[0])
    outliers_acima = int(x[x > q3 + 1.5*iqr].shape[0])
    # Observando o boxplot, vemos que há muitos outliers.
    # Talvez uma melhor alternativa seria rever a classificação de outliers.
    # Aqui acabei escolhendo remover somente se a quantidade de outliers é menor que 10%
    removeria = bool((outliers_abaixo + outliers_acima)/x.shape[0] < 0.1)
    return (outliers_abaixo, outliers_acima, removeria)


q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[15]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
vec = CountVectorizer()
X = vec.fit_transform(newsgroup['data'])
phone_id = vec.vocabulary_['phone']
X


# In[16]:


max(vec.vocabulary_.values())


# In[17]:


X[:, phone_id].sum()


# In[18]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


def q6():
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    vec = CountVectorizer()
    X = vec.fit_transform(newsgroup['data'])
    phone_id = vec.vocabulary_['phone']
    return int(X[:, phone_id].sum())


q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer


def q7():
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(newsgroup['data'])
    phone_id = tfidf.vocabulary_['phone']
    return float(np.round(sum(X[:, phone_id].toarray()), 3))


q7()

