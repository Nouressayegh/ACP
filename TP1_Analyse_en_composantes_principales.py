'''
TP 1: Analyse en composantes principales
06/10/2020
'''

import pandas as pd
import numpy as np
import copy as cp
import matplotlib.pyplot as plt

# Partie 1: programmer l'ACP sur l'espace de variables 

# Question 1:
# Visualisation de la matrice (n,p)

D = pd.read_csv("data_PDE20.txt", delimiter = " ")
print("La data frame est: \n")
print(D)

# Construction des indicateurs statistiques classiques

# La moyenne

def moyenne(df):
    n, m = df.shape
    moyenne = [sum(df.loc[:,"X{}".format(i+1)])/n for i in range(m)]
    return moyenne 

'''
m1 = moyenne(D)
print("\nLa moyenne de D est: \n" + str(m1))
'''

# La variance et l'écart-type

def variance(df):
    n, m = df.shape
    moy = moyenne(df)
    variance = [sum(df.loc[:,"X{}".format(i+1)]**2)/n - moy[i]**2 for i in range(m)]
    ecart_type = [np.sqrt(sum(df.loc[:,"X{}".format(i+1)]**2)/n - moy[i]**2) for i in range(m)]
    return variance,ecart_type

'''
v1 = variance(D)[0]
e1 = variance(D)[1]
print("\nLa variance de D est: \n" + str(v1))
print("\nL'écart-type de D est: \n" + str(e1))
'''

# La covariance

def covariance(df,i,j):
    moy = moyenne(df)
    n = df.shape[0]
    return sum((df.loc[:,"X{}".format(i+1)]- moy[i])*(df.loc[:,"X{}".format(j+1)]- moy[j]))/n

'''
cov = covariance(D, 1, 5)
print("\nLa covariance entre X1 et X5: " + str(cov)) 
'''

# Question 2

# ACP centrée

# La translation se fait en trouvant le barycentre du nuage de points qui est le vecteur 
# des moyennes  

def centrage(D):
    n,m = D.shape
    D1 = cp.deepcopy(D)
    moy = moyenne(D)
    for i in range(m):
        D1.loc[:,"X{}".format(i+1)] = D1.loc[:,"X{}".format(i+1)] - moy[i]
    
    return D1

# Verification
'''
D1 = centrage(D)   
m2 = moyenne(D1) # On remarque que les valeurs du tableau tendent vers zéro du coup, on a centré nos points dans Rp
'''

def matrice_covariance(D):
    n,m = D.shape
    Mat_cov = np.zeros((m,m))
    var = variance(D)[0]
    
    for i in range(m):
        for j in range(i,m):
            if i == j:
                Mat_cov[i][j] = var[i]
            else:
                Mat_cov[i][j] = covariance(D,i,j)
                Mat_cov[j][i] = covariance(D,i,j)
    return Mat_cov

# ACP normée

def acp_normee(df):
    d = cp.deepcopy(df)
    n,m = df.shape
    moy = moyenne(df)
    ecart = variance(df)[1]
    # Centrage
    for i in range(m):
        d.loc[:,"X{}".format(i+1)] = d.loc[:,"X{}".format(i+1)] - moy[i]
    # Réduction
    for i in range(m):
        d.loc[:,"X{}".format(i+1)] = d.loc[:,"X{}".format(i+1)] / ecart[i]
    
    return d

'''
Mat = matrice_covariance(acp_normee(D))
val_propre, vec_propre = np.linalg.eig(Mat)
vec_propre = np.transpose(vec_propre)

# L'axe factoriel de chaque hyperplan i est de vecteur normal vec_propre[i] et de valeur
# d'inertie projetée correspondante val_propre[i]

print("L'axe factoriel de l'hyperplan 1 de plus grande inertie projeté est de vecteur normal: " + str(vec_propre[0]))
print("L'inertie projetée associée est: " + str(val_propre[0]))

print("la valeur de l'inertie projetée totale est: "+ str(sum(val_propre)))
'''
# Cascade des valeurs propres

def cascade(D):

    n,m = D.shape
    Mat_cov_norm=matrice_covariance(acp_normee(D))
    val_propreN, vec_propreN = np.linalg.eig(Mat_cov_norm)
    val_propre,vec_propre = np.linalg.eig(matrice_covariance(centrage(D)))
    vec_propreN = np.transpose(vec_propreN)
    #Inertie_totaleN=sum(val_propreN)
    Inertie_cumuleeN=[]
    for i in range (m):
        val_propreNcop=cp.copy(sorted(val_propreN,reverse=True))
        Inertie_cumuleeN.append(sum(val_propreNcop[0:i]))
    
        
    plt.plot(np.arange(m),sorted(val_propreN, reverse=True))
    plt.plot(np.arange(m),sorted(val_propreN, reverse=True),'o')
    plt.title("Cascade de vecteurs propres pour une ACP normée")
    plt.ylabel("Les valeurs propres")
    plt.xlabel("le numéro des facteurs")
    plt.show() 
    plt.plot(np.arange(m),sorted(val_propre, reverse=True))
    plt.plot(np.arange(m),sorted(val_propre, reverse=True),'o')
    plt.title("Cascade de vecteurs propres pour une ACP centrée")
    plt.ylabel("Les valeurs propres")
    plt.xlabel("le numéro des facteurs")
    plt.show() 
    plt.plot(np.arange(m),Inertie_cumuleeN)
    plt.plot(np.arange(m),Inertie_cumuleeN,'o')
    plt.title("inertie cumulée pour une ACP normée")
    plt.xlabel("le numéro des facteurs")
    plt.ylabel("les inerties cumulées")
    plt.show()

# Nouvelles coordonnées:
    
def coord_Rk(D_normee,k):
    matcov = matrice_covariance(D_normee)
    val_p ,vec_p = np.linalg.eig(matcov)
    return D_normee.dot(np.transpose(vec_p[:k]))  

'''
nouveau_coord = coord_Rk(acp_normee(D),5)
nouveau_coord = nouveau_coord.loc[:8,:]
nouveau_coord.columns = ["X1","X2","X3","X4","X5"]
'''

# Qualité de la projection

def quali_representation(D_normee,coord,i):
    norme_xi = np.linalg.norm(coord.loc[i,:])
    
    M = matrice_covariance(D_normee)
    m = coord.shape[1]
    val,vec = np.linalg.eig(M)
    s = 0
    
    for j in range(m):
        axe_j = vec[j]
        axe_j = axe_j[:m]
        proj_xi_j = coord.loc[i,:].dot(np.transpose(axe_j))
        s += proj_xi_j **2
        
    return s/(norme_xi**2)
'''
l = [quali_representation(acp_normee(D),nouveau_coord,i) for i in range(nouveau_coord.shape[0])]
plt.hist(l,bins = 20)
plt.show()
'''
# Contribution des 8 individus 

def contribution_j(coord,i,j):
    n,m = coord.shape
    M = matrice_covariance(coord)
    val,vec = np.linalg.eig(M)
    axe_j = vec[j]
    proj_xi_j = coord.loc[i,:].dot(np.transpose(axe_j))
    return  (proj_xi_j ** 2)/(n*val[j])

'''
l2 = [contribution_j(nouveau_coord,i,0) for i in range(nouveau_coord.shape[0])]
'''
# Corrélation entre les p variables et p composantes
# On calcule d'abord les p composantes

nouv_comp = coord_Rk(acp_normee(D),D.shape[1])
nouv_comp.columns = ["X1","X2","X3","X4","X5","X6","X7","X8"]


def correlation(df, normee = "oui"):
    n,m = df.shape
    cor = np.zeros((m,m))
    M = matrice_covariance(df)
    ecart = variance(df)[1]
    val , vec = np.linalg.eig(M)
    vec = np.transpose(vec)
    if normee == "oui":
        for i in range(m):
            cor[:,i] = np.sqrt(abs(val[i])) * vec[i]
    else:
        for i in range(m):
            cor[:,i] = (np.sqrt(abs(val[i])) * vec[i] )/ ecart[i]
    return cor
'''
Corr = correlation(acp_normee(D), normee = "oui")
'''

# Cercle des corrélations

def cercle_corr(D,Corr):
    n,m = D.shape
    fig, axes = plt.subplots(figsize=(8,8))
    axes.set_xlim(-1,1)
    axes.set_ylim(-1,1)
    
    plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
    
    for i in range(m):
        plt.annotate(D.columns[i], (Corr[i,0],Corr[i,1]))
        
    cercle = plt.Circle((0,0),1 ,color = "blue",fill = False)
    axes.add_artist(cercle)
    plt.show()


# Comparaison avec les fonctions prédéfinies de Python
'''
# Pour créer une ACP centrée normée
# Classe pour standardisation
from sklearn.preprocessing import StandardScaler
n,m = D.shape

# Instanciation
sc = StandardScaler()

# Transformation: centrage-réduction
Z = sc.fit_transform(D)
print(Z)

# Moyenne
print(np.round(np.mean(Z,axis=0),4))
# Ecart-type
print(np.std(Z,axis=0,ddof=0))

# Classe pour l'ACP
from sklearn.decomposition import PCA

# Instanciation
ACP = PCA(svd_solver='full')

# Calculs
coord = ACP.fit_transform(Z)
# Nombre de composantes calculées
print(ACP.n_components_) 
# Variance expliquée
print(ACP.explained_variance_)
# Valeur corrigée

eigval = (n-1)/n*ACP.explained_variance_
print(eigval)

# Il faut appliquer une correction 
# Proportion de variance expliquée

print(ACP.explained_variance_ratio_)
plt.plot(np.arange(1,m+1),eigval)
plt.title("Scree plot")
plt.ylabel("Eigen values")
plt.xlabel("Factor number")
plt.show()

# Cumul de variance expliquée

plt.plot(np.arange(1,m+1),np.cumsum(ACP.explained_variance_ratio_))
plt.title("Explained variance vs. # of factors")
plt.ylabel("Cumsum explained variance ratio")
plt.xlabel("Factor number")
plt.show()

'''
# Fonction qui reclasse les valeurs propres et vecteurs propres associés en ordre croissant

def classeur(mat_cov):
    val, vec = np.linalg.eig(mat_cov)
    vec = np.transpose(vec)
    dic = {val[0]:vec[0]}
    n = val.size
    for i in range(1,n):
        dic[val[i]] = vec[i]
    val = sorted(val, reverse = True)
    i = 0
    vec2 = np.zeros((n,n))
    for key in val:
        temp = dic[key]
        for j in range(n):
            vec2[i,j] = temp[j]
        i+= 1
    return val,vec2
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        