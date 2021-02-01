import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import TP_ACP as tp1

########## Partie 1 #######

def cascade_VP(D):
    M=tp1.matrice_covariance(D)
    valp,vecp=np.linalg.eig(M)
    return sorted(valp, reverse=True)



# 1.1- Nuage isotrope

n=1000
#loi normale N(0,1)
X=np.random.randn(n)
Y=np.random.randn(n)
Z=np.random.randn(n)

X1=pd.DataFrame(data=X,columns=['X1'])
X2=pd.DataFrame(data=Y,columns=['X2'])
X3=pd.DataFrame(data=Z,columns=['X3'])

# le vecteur aléatoire
V=pd.concat([X1,X2,X3],axis=1)

'''
#moyenne
print(np.round(np.mean(Z,axis=0),4))

#écart-type
print(np.std(Z,axis=0,ddof=0))'''

# le vecteur V est centré reduit

ax = plt.axes(projection='3d')
ax.scatter3D(X, Y,Z, 'Greens')
plt.show()
Mat_cov=tp1.matrice_covariance(V)

'''
print("\n La matrice de covariance :\n"+str(Mat_cov))
#la matrice de covariance tend vers la matrice identité quand n tends vers l'infini
'''


#la cascade des valeurs propres
valeurs_p,vect_p=tp1.classeur(Mat_cov)
'''
print("\n les valeurs propres en ordre décroissant :\n"+ str(valeurs_p))'''

#la qualité de la projection
nv_coord=tp1.coord_Rk(V,2).loc[:50,:]
Q = [tp1.quali_representation(V,nv_coord,i) for i in range(nv_coord.shape[0])]
'''
plt.hist(Q)
print("\n La qualité de la projection: \n"+str(Q))'''



####1.2- Nuage non isotrope

X_ni=X
Y_ni=2*X+1
Z_ni=Z+X_ni

X1_ni=pd.DataFrame(data=X_ni,columns=['X1'])
X2_ni=pd.DataFrame(data=Y_ni,columns=['X2'])
X3_ni=pd.DataFrame(data=Z_ni,columns=['X3'])

V_ni=pd.concat([X1_ni,X2_ni,X3_ni],axis=1)

#il faut normée ce tableau
V_ni=tp1.acp_normee(V_ni)

ax = plt.axes(projection='3d')
ax.scatter3D(V_ni['X1'], V_ni['X2'],V_ni['X3'], 'Greens')
plt.show()


Mat_cov2=tp1.matrice_covariance(V_ni)
print(Mat_cov2)


#la cascade des valeurs propres
valeurs_p2,vect_p2=tp1.classeur(Mat_cov2)
'''
print("\n les valeurs propres en ordre décroissant :\n"+ str(valeurs_p2))
'''

#la qualité de la projection

nv_coord2=tp1.coord_Rk(V_ni,2).loc[:20,:]
Q2 = [tp1.quali_representation(V_ni,nv_coord2,i) for i in range(nv_coord2.shape[0])]

'''
plt.hist(Q2)
plt.show()'''

#le cas du nuage proposé

X_ni_ex=sorted(X)
Y_ni_ex=Y
Z_ni_ex=Z+np.arctan2(X_ni_ex,Y_ni_ex)

X1_ni_ex=pd.DataFrame(data=X_ni_ex,columns=['X1'])
X2_ni_ex=pd.DataFrame(data=Y_ni_ex,columns=['X2'])
X3_ni_ex=pd.DataFrame(data=Z_ni_ex,columns=['X3'])

V_ex=pd.concat([X1_ni_ex,X2_ni_ex,X3_ni_ex],axis=1)
'''
#moyenne
print("\n La moyenne de ce tableau est :\n"+ str(np.round(np.mean(V_ex,axis=0),4)))

#écart-type
print("\n l'écart type de ce tableau est :\n"+str(np.std(V_ex,axis=0,ddof=0)))
'''
#le tableau normée

V_exn=tp1.acp_normee(V_ex)

#matrice de cov non normé
Mat_cov_ex=tp1.matrice_covariance(V_ex)
'''
print("La matrice de cov non normé"+str(Mat_cov_ex))'''

#matrice de cov normé
Mat_cov_exn=tp1.matrice_covariance(V_exn)
'''
print("La matrice de cov normé"+str(Mat_cov_exn))'''

#la cascade des valeurs propres
valeurs_p3,vect_p3=tp1.classeur(Mat_cov_ex)
'''
print("\n les valeurs propres en ordre décroissant (non normée) :\n"+ str(valeurs_p3))'''

valeurs_p3n,vect_p3n=tp1.classeur(Mat_cov_exn)
'''
print("\n les valeurs propres en ordre décroissant (normée) :\n"+ str(valeurs_p3n))'''



#qualité

#1.3 point extrémaux
'''on enlève quelques enregistrements (10)'''
#pour le nuage isotrope
V=V[:n-10]
valeurs_p=cascade_VP(V)

print("\n Les valeurs propres en ordre décroissant :\n"+ str(valeurs_p))
''' le valeurs propres ne changes pas'''

#pour le nuage non isotrope
V_ex=V_ex[:n-10]

valeurs_p=cascade_VP(V_ex)
print("\n Les valeurs propres en ordre décroissant pour le nuage non normé:\n"+ str(valeurs_p))


V_exn=V_exn[:n-10]
valeurs_p=cascade_VP(V_exn)
print("\n Les valeurs propres en ordre décroissant  pour le nuage non isotrope normé:\n"+ str(valeurs_p))

#il y a une difference entre les valeurs propres ! donc l'ACP peut être mal interpretée 
"""
####### Partie 2 #####

#nuage isotrope
''' ACP sur R^n'''

Mp=tp1.matrice_covariance(V)
valp_p,vecp_p=np.linalg.eig(Mp)
vecp_p = np.transpose(vecp_p)

V=V.transpose()
Mn=tp1.matrice_covariance(V)
valp_n,vecp_n=np.linalg.eig(Mn)
vecp_n = np.transpose(vecp_n)

"""
#vecp_n_exp=1/np.sqrt(valp_n)*np.dot(V,vecp_p)