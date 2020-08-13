# AceleraDev - Data Science

O AceleraDev - Data Science foi um programa de aceleração em Data Science realizado pela [Codenation](https://www.codenation.dev/).

Os participantes têm acesso a video-aulas e outros conteúdos e, semanalmente, resolvem desafios propostos para aplicar os conhecimentos adquiridos naquela semana.


## Este repositório

Aqui estão disponibilizadas as minhas soluções para os desafios semanais com uma breve descrição.

Cada um dos desafios acompanha um README contendo instruções para a realização do desafio.  
Muitos dos desafios são propostos/realizados em Jupyter notebooks. A grande maioria destes estarão nomeados como "main.ipynb".  
Alguns dos README contém apenas instruções para obtenção dos dados. Nesses casos, as questões estão no próprio Jupyter notebook do desafio.

Onde for relevante, os datasets são manipulados utilizando Python Pandas, Numpy, Scikit-learn e Scipy.stats.

## Por conteúdo
### Noções de estatística
- [Conhecendo melhor nossa base de consumidores: qual estado possui os clientes com melhores pontuações de crédito?](coestatistica-1)
    - Média, mediana, moda, desvio padrão, agregação de dados.
    
- [Funções de probabilidade](data-science-1)
    - Distribuições de probabilidade (normal e binomial), função de distribuição acumulada (CDF), CDF empírica (ECDF), quantis, padronização.
    
- [Funções de probabilidade 2](data-science-2)
    - Testes de hipóteses (Shapiro-Wilk, Jarque-Bera, D'Agostino-Pearson), qq-plots.
    
### Pré-processamento de dados
- [Redução de dimensionalidade e seleção de variáveis](data-science-3)
    - Redução de dimensionalidade, PCA, seleção de variáveis, RFE.
    
- [Feature engineering](data-science-4)
    - Processamento de texto, binning, one-hot encoding, pipelines, imputação, detecção de outliers.
    
### Juntando tudo
- [Descubra as melhores notas de matemática do ENEM 2016](enem-2)
    - Análise exploratória de dados (EDA), pré-processamento dos dados, feature engineering, clusterização (k-Means), visualização interativa de dados, regressão, GridSearch.
    - Neste notebook, algumas visualizações interativas não são renderizadas pelo GitHub. Você pode [visualizá-las aqui](https://nbviewer.jupyter.org/github/tkleyton/codenation/blob/master/enem-2/main.ipynb)
    
- [Descubra quem fez o ENEM 2016 apenas para treino](enem-4)
    - Análise exploratória de dados (EDA), pré-processamento dos dados, feature engineering, tratamento de dados desbalanceados (SMOTE), modelos de classificação, modelo de votação de classificadores.
    
- [Recomendação de Leads (Projeto Final)](projeto_final)
    - Pode ser dividido em 3 etapas:
        1. [Pré-processamento dos dados](projeto_final/preprocessing.ipynb)
        2. [Aplicação do modelo de recomendação](projeto_final/model.ipynb)
        3. [Visualização dos resultados](projeto_final/visualization.ipynb)
            - Aqui também há o problema da renderização de gráficos interativos no GitHub. Você pode visualizar o notebook [aqui](https://nbviewer.jupyter.org/github/tkleyton/codenation/blob/master/projeto_final/visualization.ipynb).
            - Alternativamente, realizei o deploy destes mesmos gráficos em um [Dash no Heroku](https://leads-generator-dash.herokuapp.com/). O código-fonte pode ser encontrado [aqui](https://github.com/tkleyton/aceleradev-leads-dash) 
