import pandas as pd 
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

df=pd.read_excel("calisan_verisi.xlsx")


df=df.drop(["Calisan ID","unvan"],axis=1)

X=df[["UnvanSeviyesi","Kidem","Puan"]]
y=df[["maas"]]
print(df.corr())
#veriler arasında bir ilişkinin olup olmadığına bakıyoruz 1e ne kadar yakınsa ilişki güçlü uzaksa ilişki zayıf eksi ise hiç bir ilişki yok anlamına geliyor.

# print(df)
lin_reg=LinearRegression()
lin_reg.fit(X,y)
tahmin=lin_reg.predict(X)

x_sabit_terim=sm.add_constant(X)
model=sm.OLS(y,x_sabit_terim).fit()
model_tahmin=model.predict(x_sabit_terim)
print("OLS sonuçları: ",model.summary())
("-------------------------------------------------------------------------------------------------------------------------------------------------")
# plt.plot(y, label="maaş")
# plt.plot(model_tahmin, label="OLS tahmin",marker="o")
# plt.xlabel("çalışan")
# plt.ylabel("maaş")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
print("-------------------------------------------------------------------------------------------------------------------------------------------------")

poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
tahmin2=lin_reg2.predict(x_poly)
model2=sm.OLS(tahmin2,X).fit()
print("POLY model sonuçları: ",model2.summary())

scaler=StandardScaler()
x_olcekli=scaler.fit_transform(X)
y_olcekli=scaler.fit_transform(y)

("-------------------------------------------------------------------------------------------------------------------------------------------------")
svr_reg=SVR(kernel="rbf")
y_olcekli=y_olcekli.ravel()
svr_reg.fit(x_olcekli,y_olcekli)
tahmin3=svr_reg.predict(x_olcekli)
model3=sm.OLS(tahmin3,x_olcekli).fit()
print("SVR model sonuçları: ",model3.summary())
print("-------------------------------------------------------------------------------------------------------------------------------------------------")

d_tree=DecisionTreeRegressor()
d_tree.fit(X,y)
tahmin4=d_tree.predict(X)
model4=sm.OLS(tahmin4,X).fit()
print("DecisionTreeRegressor model sonuçları: ",model4.summary())
print("-------------------------------------------------------------------------------------------------------------------------------------------------")


random=RandomForestRegressor(n_estimators=10,random_state=42)
random.fit(X,y)
tahmin5=random.predict(X)
model5=sm.OLS(tahmin5,X).fit()
print("RandomForestRegressor model sonuçları: ",model5.summary())
("-------------------------------------------------------------------------------------------------------------------------------------------------")