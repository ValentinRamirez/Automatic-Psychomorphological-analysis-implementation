import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut
import numpy as np

X=np.array([0.535535714286, 0.538094024717, 0.45564281559, 0.547798066595, 0.491280986287, 0.482575757576, 0.525419446129,0.433437569801, 0.509, 0.502966117116, 0.437310606061, 0.499810606061,
   0.447529644269,0.482049213392, 0.4904223854, 0.501418439716, 0.429341963323,  0.570832202009, 0.443299869053, 0.442071178754, 0.483991964195,
   0.472333333333, 0.54898989899, 0.484531450578, 0.518229376258, 0.437910733844, 0.558069729855, 0.54171539961, 0.469164612986, 0.425728319783, 0.553795834051, 0.527255639098, 0.54183323815, 0.376006261181,
   0.487306769723, 0.549611946164, 0.457857142857, 0.519351935194, 0.507208840542, 0.594060606061, 0.47243513314, 0.521739130435, 0.555297644954,  0.534567901235,
   0.503132284382, 0.594132226952])

Y=np.array([8,6,3,9,8,4,7,4,8,6,4,5,6,4,6,4,5,9,5,3,8,4,7,8,4,5,7,6,6,5,8,6,8,3,5,6,7,8,7,9,5,8,8,5,7,8])

#print X.shape,Y.shape


#print len(X), len(Y)
plt.figure()
plt.title('Expansion/Retraccion')
plt.scatter(X,Y,c='r')
plt.show()


V=np.array([0.3977376906,0.3859248405,0.3960493236,0.3277682298,0.4416689499,0.4288526179,0.4535921539,0.3628671403,0.3878022613,0.3584943884,0.477809254,0.4853551368,0.4401506093,0.466569369698,0.3990224281,0.3462202278,
   0.3694692307,0.3963622481,0.4172173956,0.4551239719,0.3871699391,0.3920585241,0.428876981,0.3784865736,0.399519548,0.3897163736,0.4861602709,0.4442270949,0.3452397182,0.425573028,0.3557058116,0.3569001191,0.4191616766,
   0.4301860892,0.408993514, 0.3918735596,0.417652647,0.3449357165,0.4725878874,0.3962148386,0.4197689079,0.4577682188,0.338579795,0.507098491571,0.420545091385,0.447402258905,0.391716882512,0.384048830112,
   0.383111352723])

W=np.array([5,7,7,4,7,8,7,3,8,3,7,8,7,8,4,4,5,3,6,7,3,5,6,7,6,3,8,6,5,7,6,4,6,6,8,6,7,3,7,5,5,8,4,8,5,6,6,4,5])

plt.figure()
plt.title('Triangle_of_senses')
plt.scatter(V,W,c='r')
#plt.title('Ratio Expansion/Retraccion')
#plt.scatter(X2,Y,c='r')
plt.show()

loo = LeaveOneOut()
regr = linear_model.LinearRegression()

msq=np.zeros(46)
coef=np.zeros(46)
inter=np.zeros(46)
i=0
for train, test in loo.split(X):
      # Train the model using the training sets
        X_train=X[train].reshape(-1,1)
        Y_train=Y[train].reshape(-1,1)
        X_test=X[test].reshape(-1,1)
        Y_test=Y[test].reshape(-1,1)
        regr.fit(X_train, Y_train)
        msq[i]=(regr.predict(X_test)-Y_test)**2
        coef[i]=regr.coef_
        inter[i]=regr.intercept_
        i+=1
error=np.mean(msq)
A=np.mean(coef)
b=np.mean(inter)

print '\nAverage Mean sqared error Exp/Ret=',error,' A= ', A,'b=', b

msq2=np.zeros(46)
coef2=np.zeros(46)
inter2=np.zeros(46)
i=0
for train, test in loo.split(X):
      # Train the model using the training sets
        V_train=V[train].reshape(-1,1)
        W_train=W[train].reshape(-1,1)
        V_test=V[test].reshape(-1,1)
        W_test=W[test].reshape(-1,1)
        regr.fit(V_train, W_train)
        msq2[i]=(regr.predict(V_test)-W_test)**2
        coef2[i]=regr.coef_
        inter2[i]=regr.intercept_
        i+=1
error2=np.mean(msq2)
A2=np.mean(coef2)
b2=np.mean(inter2)

print '\nAverage Mean sqared error Triangle_of_senses=',error2,' A= ', A2,'b=', b2


plt.figure()
plt.title('Expansion/Retraction')
plt.scatter(X, Y,  color='red')
plt.plot(X, A*X+b, color='blue',linewidth=3)
plt.show()

plt.figure()
plt.title('Triangle_of_senses')
plt.scatter(V, W,  color='red')
plt.plot(V, A2*V+b2, color='blue',linewidth=3)
plt.show()
