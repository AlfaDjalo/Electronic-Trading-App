from models import ModelHandler
from StockData import StockData

class Comparison:
    def __init__(self, stock, model, train_date='2022-12-31'):
        self.ticker = stock
        self.model = None

 
