import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#PROBLEMATIQUE
# quels sont les facteurs qui peuvent influencer les resultats des etudiants ?

#valeurs = pd.read_csv(r"C:\Users\fnoal\OneDrive - etu.utc.fr\UTC\4 - GI04\SY09\Projet\student-mat.csv", header=0, sep=',')

"""
Medu : niveau d'éducation de la mère (0 à 4)
Fedu : niveau d'éducation du père (0 à 4)
Mjob : profession de la mere ()
Fjob : profession du pere
Dalc : consommation d'alcool en un jour (de 1 à 5)
Walc : consommation d'alcool en un week-end (de 1 à 5)
G1 : 3 à 19
G2 : 0 à 19
G3 : 0 à 20
Pstatus : statut de cohabitation des parents (T = together, A = alone)
"""

d1=pd.read_csv(r"C:\Users\fnoal\OneDrive - etu.utc.fr\UTC\4 - GI04\SY09\Projet\student-mat.csv",sep=",",header=0)
d2=pd.read_csv(r"C:\Users\fnoal\OneDrive - etu.utc.fr\UTC\4 - GI04\SY09\Projet\student-por.csv",sep=",",header=0)

d3=pd.merge(d1,d2, on=["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet", "studytime", "schoolsup", "famsup", "activities", "guardian"], how= 'outer', suffixes= ("_mat","_por"))
print(d3.shape[0])
d3.info()

d3['G_mat'] = d3[['G1_mat', 'G2_mat', 'G3_mat']].mean(axis=1)

plt.figure(figsize=(10, 6))
d3.boxplot(column='G_mat', by='age')
plt.suptitle('')
plt.title('Distribution de la moyenne des notes de maths par âge')
plt.xlabel('Âge')
plt.ylabel('Moyenne mathématiques (G1, G2, G3)')
plt.show()

print(d3.groupby('age')['G_mat'].mean())

