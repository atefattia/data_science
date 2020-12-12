class BaseModel(ABC):
    @classmethod
    @abstractmethod
    def load(cls, model_path: str):
        pass

    @abstractmethod
    def save(self, model_path: str):
        pass

    @abstractmethod
    def fit(self, x_train: list, y_train: list = None):
        pass

    @abstractmethod
    def metrics(self, test_io: list, test_nio: list):
        pass

    @abstractmethod
    def predict_files(self, file_paths: list):
        pass
