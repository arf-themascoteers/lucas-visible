

def x(z,a,b=None):
    print(z, a, b)

def y(c):
    if isinstance(c, str):
        x("dss","hello",c)
    if type(c) is dict:
        x("ddd",**c)

y("h")
y({"a":"ni", "b":"hao"})
