class LinearRegression:

    def fit(self, X, y):
        
        n = len(X)

        mean_x = sum(X) / n
        mean_y = sum(y) / n

        numerator = 0
        denominator = 0

        for i in range(n):
            numerator += (X[i] - mean_x) * (y[i] - mean_y) 
            denominator += (X[i] - mean_x) ** 2

        self.a = numerator / denominator
        self.b = mean_y - self.a * mean_x

    def predict(self, X):

        predictions = []

        for x in X:
            y = self.a * x + self.b
            predictions.append(y)

        return predictions