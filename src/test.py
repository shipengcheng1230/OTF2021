def f(x, y):
    if x == y:
        return x
    else:
        if x > y:
            return f(x - y, y)
        else:
            return f(x, y - x)

print(f(2437, 875))