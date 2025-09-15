class DataSplitter:
    def __init__(self, df, train_ratio=0.8, val_ratio=0.1):
        self.df = df
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
    
    def split(self):
        n = len(self.df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        train = self.df[:train_end]
        val = self.df[train_end:val_end]
        test = self.df[val_end:]
        return train, val, test
