

class Module:
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
        f'{self.__class__.__name__}.forward() not implemented'
        )