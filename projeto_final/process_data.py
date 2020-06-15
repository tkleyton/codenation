import pandas as pd
from sklearn.decomposition import PCA


def merge_portolio_ids_to_base(portfolio_ids, base):
    return portfolio_ids.merge(base, on='id', how='inner')

# ---------------------------------------------------------------
# Preparing data for barplots
n_of_portfolios = 3
ports_ids = [pd.read_csv(f'data/port_ids{i}.csv', index_col='id') for i in range(1, n_of_portfolios + 1)]
leads_ids = [pd.read_csv(f'data/leads_ids{i}.csv', index_col='id') for i in range(1, n_of_portfolios + 1)]

# Filter the first n leads, n being the size of the portfolio.
leads_ids = [leads_ids[i].iloc[:ports_ids[i].shape[0]] for i in range(n_of_portfolios)]

df_orig = pd.read_csv('data/estaticos_market.csv', index_col='id')
ports_dfs = [merge_portolio_ids_to_base(port_ids, df_orig) for port_ids in ports_ids]
leads_dfs = [merge_portolio_ids_to_base(lead_ids, df_orig) for lead_ids in leads_ids]

for i in range(3):
    ports_dfs[i].to_csv(f'data/ports_dfs{i+1}.csv')
    leads_dfs[i].to_csv(f'data/leads_dfs{i+1}.csv')

# ---------------------------------------------------------------
# Preparing data for scatterplot
df_clean = pd.read_csv('data/estaticos_market_clean.csv', index_col='id')
pca = PCA(n_components=2)
reduced_df = pca.fit_transform(df_clean)
reduced_df = pd.DataFrame(data=reduced_df, columns=['x', 'y'], index=df_clean.index)
reduced_df.to_csv('data/reduced_df.csv')

ports_dfs_clean = [merge_portolio_ids_to_base(port_ids, reduced_df) for port_ids in ports_ids]
leads_dfs_clean = [merge_portolio_ids_to_base(lead_ids, reduced_df) for lead_ids in leads_ids]

for i in range(3):
    ports_dfs_clean[i].to_csv(f'data/ports_dfs_clean{i+1}.csv')
    leads_dfs_clean[i].to_csv(f'data/leads_dfs_clean{i+1}.csv')
