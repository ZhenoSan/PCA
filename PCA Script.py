import pandas as pd
from numpy import linalg as LA

df = pd.read_csv('cars.csv')
MeanCenteredDf = df.iloc[:,3:].apply(lambda x: x-x.mean())

cov_mat = MeanCenteredDf.cov()
corr_mat = MeanCenteredDf.corr()

values, vectors = LA.eig(cov_mat)

corr_e_values, corr_e_vectors = LA.eig(corr_mat)

ProjDf = MeanCenteredDf.dot(vectors)

ProjDf = pd.DataFrame(data = ProjDf.iloc[:, :2]).add_prefix('principal component ')
ProjDf = pd.concat([ProjDf, df['Type']], axis = 1)

ProjDf_corr = MeanCenteredDf.dot(corr_e_vectors)

ProjDf_corr = pd.DataFrame(data = ProjDf_corr.iloc[:, :2]).add_prefix('principal component ')
ProjDf_corr = pd.concat([ProjDf_corr, df['Type']], axis = 1)

fig = plt.figure(figsize = (15,5))
ax = fig.add_subplot(1,2,1) 

ax.set_xlabel('Principal Component 1', fontsize = 10)
ax.set_ylabel('Principal Component 2', fontsize = 10)
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
Types = ['Small', 'Medium', 'Large', 'Sporty', 'Compact', 'Van']
colors = ['r', 'g', 'b', 'y', 'c', 'm']
for Type, color in zip(Types,colors):
    indicesToKeep = ProjDf_corr['Type'] == Type
    ax.scatter(ProjDf_corr.loc[indicesToKeep, 'principal component 0']
               , ProjDf_corr.loc[indicesToKeep, 'principal component 1']
               , c = color
               , s = 50)
ax.legend(Types,fontsize = 'small')
ax.grid()

ax2 = fig.add_subplot(1,2,2) 
ax2.set_xlabel('Principal Component 1', fontsize = 10)
ax2.set_ylabel('Principal Component 2', fontsize = 10)
ax2.xaxis.set_tick_params(labelsize=10)
ax2.yaxis.set_tick_params(labelsize=10)
for Type, color in zip(Types,colors):
    indicesToKeep = ProjDf['Type'] == Type
    ax2.scatter(ProjDf.loc[indicesToKeep, 'principal component 0']
               , ProjDf.loc[indicesToKeep, 'principal component 1']
               , c = color
               , s = 50)
#ax2.legend(Types,fontsize = 'small')
ax2.grid()