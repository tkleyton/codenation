import pandas as pd


# Costumo colocar a coluna de id como index do DataFrame
df = pd.read_csv('desafio1.csv', index_col='id')

answer = df.groupby(['estado_residencia']).\
    agg({
        'pontuacao_credito': [
            pd.Series.mode, 'median',
            'mean','std']
        })

answer.columns = [
    'moda',
    'mediana',
    'media',
    'desvio_padrao']

# Para que o JSON esteja no formato da submissão 
# deve-se orientar por linhas, não colunas, que é o padrão
answer.to_json('submission.json', orient='index')
