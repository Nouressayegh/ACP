'''
TP 2: Analyse en composantes principales partie 2
07/10/2020
'''


import TP1_Analyse_en_composantes_principales as tp
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

# Partie 1: ACP et étude de nuage de points
# Etude de la forme du nuage initiale et de sa répercussion sur la réduction de dimension

# Nuage isotrope

n = 400
X = np.random.randn(n)
Y = np.random.randn(n)
Z = np.random.randn(n)

vecteur = (X,Y,Z)
vecteur = vecteur/np.linalg.norm(vecteur)
vecteur = np.transpose(vecteur)

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.set_xlabel('X Axes')
ax.set_ylabel('Y Axes')
ax.set_zlabel('Z Axes')

ax.scatter(vecteur[:,0],vecteur[:,1],vecteur[:,2])
plt.show()

D = pd.DataFrame(vecteur)
D.columns = ["X1","X2","X3"]

# Mise au point de l'ACP

# Centrage et réduction de la data frame

D_norm = tp.acp_normee(D)
ax.scatter(D_norm.loc[:,"X1"],D_norm.loc[:,"X2"],D_norm.loc[:,"X3"],color="green")
plt.show()

# Matrice de variance-covariance, valeurs propres et vecteurs propres

Mat_cov = tp.matrice_covariance(D_norm)
val_propre, vec_propre = tp.classeur(Mat_cov)

# Cascade des valeurs propres

tp.cascade(D)

# On voit bien que les 3 valeurs sont supérieurs à 1 et sont très proches. Selon les deux critères
# vus en cours, on ne peut pas projeter sur un espace de dimension plus petite tout en conservant
# l'information

# Nouvelles coordonées
'''

nouveau_coord = tp.coord_Rk(D_norm,2)
nouveau_coord.columns = ["X1","X2"]
plt.scatter(nouveau_coord.loc[:,"X1"],nouveau_coord.loc[:,"X2"], color = "orange")
plt.title("Projection de la data frame sur un plan 2D")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Qualité de la projection 

l = [tp.quali_representation(D_norm,nouveau_coord,i) for i in range(nouveau_coord.shape[0])]
plt.hist(l)
plt.show()
'''

# On sait que plus l’angle est proche de 0 (et donc le cos proche de1) plus l’individu est 
# bien projeté. On voit bien sur l'histogramme que les individus mal projetés sont plus nombreux
# que les individus bien projetés. Ce qui prouve notre conclusion de tout à l'heure.

# Cercle des corrélations

Cor = tp.correlation(D_norm,"oui")
tp.cercle_corr(D,Cor)

# l'angle séparant X1 et X2 et l'angle séparant X1 et X3 sont proche de 90°, elles sont alors 
# deux à deux indépendantes.


# Nuage anisotrope

X_ani = X
Y_ani = 2*X + 1
Z_ani = Z + X_ani

vecteur2 = (X_ani,Y_ani,Z_ani)
vecteur2 = vecteur2/np.linalg.norm(vecteur2)
vecteur2 = np.transpose(vecteur2)

D_ani = pd.DataFrame(vecteur2)
D_ani.columns = ["X1","X2","X3"]
D_ani_norm = tp.acp_normee(D_ani)

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.set_xlabel('X Axes')
ax.set_ylabel('Y Axes')
ax.set_zlabel('Z Axes')

ax.scatter(vecteur2[:,0],vecteur2[:,1],vecteur2[:,2])
plt.show()


# Mise au point de l'ACP
# Matrice de variance_coavriance, valeurs propres et vecteurs propres

Mat_cov2 = tp.matrice_covariance(D_ani_norm)
val_propre2, vec_propre2 = tp.classeur(Mat_cov2)

# Cascade des valeurs propres

tp.cascade(D_ani_norm)

# On voit que les deux dernières valeurs propres sont trsè faibles devant la première, et d'après
# le critère de Karlis - Saporta - Spinaki, on prend les valeurs propres supérieures à
# 0.0894 et par conséquent, on garde les deux premières.

# Qualité de projection 

nv_coord = tp.coord_Rk(D_ani_norm,2)
nv_coord.columns = ["X1","X2"]
quali = [tp.quali_representation(D_ani_norm,nv_coord,i) for i in range(nv_coord.shape[0])]

plt.scatter(nv_coord.loc[:,"X1"],nv_coord.loc[:,"X2"], color = "green")
plt.title("Projection sur le plan (X,Y)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Cercle des corrélations

Cor2 = tp.correlation(D_ani_norm,"oui")
tp.cercle_corr(D_ani,Cor2)

# On voit clairement que X1 et X2 sont superposés, l'angle entre est donc 0 et ils sont positivement
# corrélés, X1 et X3 sont colinéaire et donc très fortement corrélés.

# Points extrémaux

point = (np.max(vecteur2[:,0]),np.max(vecteur2[:,1]),np.max(vecteur2[:,2]))
point_op = (np.min(vecteur2[:,0]),np.min(vecteur2[:,1]),np.min(vecteur2[:,2]))
vecteur2 = vecteur2.tolist()
for i in range(10):
    vecteur2.append(point)
    vecteur2.append(point_op)
D_alongee = pd.DataFrame(vecteur2)

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.set_xlabel('X Axes')
ax.set_ylabel('Y Axes')
ax.set_zlabel('Z Axes')

ax.scatter(D_alongee.loc[:,0],D_alongee.loc[:,1],D_alongee.loc[:,2])
plt.show()

# Partie 2: Etude de la forme du nuage initiale sur la réduction de dimension dans les deux espaces
'''
# Cas 1: Nuage isotrope

# Question 1: Mettons en oeuvre l'ACP sur Rn

def classer(val,vec):
    val = val.real
    vec = vec.real
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
    return np.array(val),np.array(vec2)
        
# Cas 1: nuage isotrope

XX_prime = D_norm.dot(np.transpose(D_norm))
XX_prime.columns = ["X{}".format(i+1) for i in range(XX_prime.shape[0])]
val_propre_Rn, vec_propre_Rn = np.linalg.eig(XX_prime)
vec_propre_Rn = np.transpose(vec_propre_Rn)
val_propre_Rn,vec_propre_Rn = classer(val_propre_Rn,vec_propre_Rn)
m = XX_prime.shape[0]

# Cascade des valeurs propres

plt.plot(np.arange(m),sorted(val_propre_Rn, reverse=True))
plt.plot(np.arange(m),sorted(val_propre_Rn, reverse=True),'o')
plt.title("Cascade de vecteurs propres pour une ACP normée")
plt.ylabel("Les valeurs propres")
plt.xlabel("le numéro des facteurs")
plt.show() 

# D'après le critère de Karlis - Saporta - Spinaki, on ne prend que les valeurs propres supérieurs
# ou égales à 9.89 et donc les trois premières valeurs propres


# Qualité de projection

nouvelle_coord = (np.transpose(D_norm)).dot(np.transpose(vec_propre_Rn[:3])) 
nouvelle_coord.columns = ["X1","X2","X3"]
nv_c = np.zeros((3,3))

for i in range(3):
    for j in range(3):
        nv_c[i,j] = nouvelle_coord.loc["X{}".format(i+1),"X{}".format(j+1)].real
        
nv_c = pd.DataFrame(nv_c)
nv_c.columns = ["X1","X2","X3"]


fig = plt.figure()
ax = plt.axes(projection="3d")

ax.set_xlabel('X Axes')
ax.set_ylabel('Y Axes')
ax.set_zlabel('Z Axes')

ax.scatter(nv_c.loc[0,:],nv_c.loc[1,:],nv_c.loc[2,:])
plt.show()

def quali(nouvelle_coord,vec_propre_Rn,i):

    norme_xi = np.linalg.norm(nouvelle_coord.loc["X{}".format(i+1),:])
    m = nouvelle_coord.shape[1]
    s = 0
    
    for j in range(m):
        axe_j = vec_propre_Rn[j]
        axe_j = axe_j[:m]
        proj_xi_j = nouvelle_coord.loc["X{}".format(i+1),:].dot(np.transpose(axe_j))
        s += proj_xi_j **2
        
    return s/(norme_xi**2)

qualite = [quali(nouvelle_coord,vec_propre_Rn,i) for i in range(nouvelle_coord.shape[0])]

# Cercle des corrélations

cor = np.zeros((m,m))

for i in range(m):
    cor[:,i] = np.sqrt(abs(val_propre_Rn[i])) * vec_propre_Rn[i]
    
fig, axes = plt.subplots(figsize=(5,5))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)

plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)

for i in range(m):
    plt.annotate(XX_prime.columns[i], (cor[i,0],cor[i,1]))
    
cercle = plt.Circle((0,0),1 ,color = "blue",fill = False)
axes.add_artist(cercle)
plt.show()

# Cas 2: nuage anisotrope

XX_prime2 = D_ani_norm.dot(np.transpose(D_ani_norm))
XX_prime2.columns = ["X{}".format(i+1) for i in range(XX_prime2.shape[0])]
val_propre_Rn2, vec_propre_Rn2 = np.linalg.eig(XX_prime2)
vec_propre_Rn2 = np.transpose(vec_propre_Rn2)
val_propre_Rn2,vec_propre_Rn2 = classer(val_propre_Rn2,vec_propre_Rn2)
m = XX_prime2.shape[0]

# Cascade des valeurs propres

plt.plot(np.arange(m),sorted(val_propre_Rn2, reverse=True))
plt.plot(np.arange(m),sorted(val_propre_Rn2, reverse=True),'o')
plt.title("Cascade de vecteurs propres pour une ACP normée")
plt.ylabel("Les valeurs propres")
plt.xlabel("le numéro des facteurs")
plt.show() 

# D'après le critère de Karlis - Saporta - Spinaki, on ne prend que les valeurs propres supérieurs
# ou égales à 9.89 et donc les 2 premières valeurs propres


# Qualité de projection

nouvelle_coord2 = (np.transpose(D_ani_norm)).dot(np.transpose(vec_propre_Rn2[:2])) 
nouvelle_coord2.columns = ["X1","X2"]
nv_c2 = np.zeros((3,2))

for i in range(3):
    for j in range(2):
        nv_c2[i,j] = nouvelle_coord2.loc["X{}".format(i+1),"X{}".format(j+1)].real
        
nv_c2 = pd.DataFrame(nv_c2)
nv_c2.columns = ["X1","X2"]

plt.scatter(nv_c2.loc[0,:],nv_c2.loc[1,:],nv_c2.loc[2,:])
plt.show()

qualite2 = [quali(nouvelle_coord2,vec_propre_Rn2,i) for i in range(nouvelle_coord2.shape[0])]

# Cercle des corrélations

cor2 = np.zeros((m,m))
va,ve = np.linalg.eig(XX_prime2)
ve = np.transpose(ve)

for i in range(m):
    cor2[:,i] = np.sqrt(abs(va[i])) * ve[i]
    
fig, axes = plt.subplots(figsize=(5,5))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)

plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)

for i in range(m):
    plt.annotate(XX_prime2.columns[i], (cor2[i,0],cor2[i,1]))
    
cercle2 = plt.Circle((0,0),1 ,color = "blue",fill = False)
axes.add_artist(cercle2)
plt.show()
'''
# Question 2: Cf rapport

# Question 3 et 4:
'''
u_alpha = []

for i in range(vec_propre.shape[0]):
    u_alpha.append(np.transpose(D_norm).dot(np.transpose(vec_propre_Rn[i]))/np.sqrt(val_propre_Rn[i]))


T = [[0,0,0] for i in range(3)]
for i in range(3):
    for j in range(3):
        T[i][j] = u_alpha[i][j]
    
'''

# Question 5: Effet de la dimension sur les deux décompositions
'''
# On va essayer ici d'augementer le nombre d'individus.

# Cas du nuage isotrope

p = 20
vecteur3 = []
for i in range(p):
    vecteur3.append(np.random.randn(n))

vecteur3 = vecteur3 / np.linalg.norm(vecteur3)
vecteur3 = np.transpose(vecteur3)

D2 = pd.DataFrame(vecteur3)
D2.columns = ["X{}".format(i+1) for i in range(p)]

D_norm2 = tp.acp_normee(D2)

XX_prime_dim = D_norm2.dot(np.transpose(D_norm2))
XX_prime_dim.columns = ["X{}".format(i+1) for i in range(XX_prime_dim.shape[0])]
val_propre_Rn_dim, vec_propre_Rn_dim = np.linalg.eig(XX_prime_dim)
vec_propre_Rn_dim = np.transpose(vec_propre_Rn_dim)
val_propre_Rn_dim,vec_propre_Rn_dim = classer(val_propre_Rn_dim,vec_propre_Rn_dim)
m = XX_prime_dim.shape[0]

# Cascade des valeurs propres

plt.plot(np.arange(m),sorted(val_propre_Rn_dim, reverse=True))
plt.plot(np.arange(m),sorted(val_propre_Rn_dim, reverse=True),'o')
plt.title("Cascade de vecteurs propres pour une ACP normée")
plt.ylabel("Les valeurs propres")
plt.xlabel("le numéro des facteurs")
plt.show() 

# Nouvelles coordonnées et qualité de projection

nouvelle_coord_dim = (np.transpose(D_norm2)).dot(np.transpose(vec_propre_Rn_dim[:20])) 
nouvelle_coord_dim.columns = ["X{}".format(i+1) for i in range(p)]
qualite_dim = [quali(nouvelle_coord_dim,vec_propre_Rn_dim,i) for i in range(nouvelle_coord_dim.shape[0])]
plt.hist(qualite_dim)
plt.show()

# Cas du nuage anisotrope

p = 20
vecteur4 = [np.random.randn(n)]
for i in range(1,int(p/2)):
    vecteur4.append(vecteur4[i-1]*i)
    
for i in range(int(p/2),p):
    vecteur4.append(np.random.randn(n)+ vecteur4[i-int(p/2)])
    
vecteur4 = vecteur4/np.linalg.norm(vecteur4)
vecteur4 = np.transpose(vecteur4)
D3 = pd.DataFrame(vecteur4)
D3.columns = ["X{}".format(i+1) for i in range(p)]

D_norm3 = tp.acp_normee(D3)

XX_prime2_dim = D_norm3.dot(np.transpose(D_norm3))
XX_prime2_dim.columns = ["X{}".format(i+1) for i in range(XX_prime2_dim.shape[0])]
val_propre_Rn2_dim, vec_propre_Rn2_dim = np.linalg.eig(XX_prime2_dim)
vec_propre_Rn2_dim = np.transpose(vec_propre_Rn2_dim)
val_propre_Rn2_dim,vec_propre_Rn2_dim = classer(val_propre_Rn2_dim,vec_propre_Rn2_dim)
m = XX_prime2_dim.shape[0]

# Cascade des valeurs propres

plt.plot(np.arange(m),sorted(val_propre_Rn2_dim, reverse=True))
plt.plot(np.arange(m),sorted(val_propre_Rn2_dim, reverse=True),'o')
plt.title("Cascade de vecteurs propres pour une ACP normée")
plt.ylabel("Les valeurs propres")
plt.xlabel("le numéro des facteurs")
plt.show() 

# Nouvelles coordonnées et qualité de projection

nouvelle_coord2_dim = (np.transpose(D_norm3)).dot(np.transpose(vec_propre_Rn2_dim[:4])) 
nouvelle_coord2_dim.columns = ["X{}".format(i+1) for i in range(4)]
qualite2_dim = [quali(nouvelle_coord2_dim,vec_propre_Rn2_dim,i) for i in range(nouvelle_coord2_dim.shape[0])]
plt.hist(qualite2_dim)
plt.show()

'''

