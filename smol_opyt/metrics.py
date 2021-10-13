def accuracy(y, yhat):
    n = len(y)
    return sum(y == yhat) / n
