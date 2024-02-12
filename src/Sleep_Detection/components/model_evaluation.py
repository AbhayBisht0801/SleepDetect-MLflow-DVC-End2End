import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from src.Sleep_Detection.config.configuration import EvaluateConfig
from src.Sleep_Detection.utils.common import save_json

os.environ['MLFLOW_TRACKING_URI']='https://dagshub.com/AbhayBisht0801/SleepDetect-MLflow-DVC-End2End.mlflow' 
os.environ['MLFLOW_TRACKING_USERNAME']='AbhayBisht0801'
os.environ['MLFLOW_TRACKING_PASSWORD']='dae5a0dsdsd2c6d887749ffe32bcacasa2454512451646464'

class Evaluation:
    def __init__(self,config=EvaluateConfig):
        self.config=config
    def test_generator(self):
        datagenerator = dict(
            rescale = 1./255
            
        )

        dataflow = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator
        )

        self.test_generator = test_datagenerator.flow_from_directory(
            directory=self.config.testing_data,
            shuffle=False,
            **dataflow
        )
    @staticmethod
    def load_model(path:Path)->tf.keras.Model:
        return tf.keras.models.load_model(path)
    def evaluation(self):
        self.model=self.load_model(self.config.path_of_model)
        self.test_generator()
        self.score=self.model.evaluate(self.test_generator)
    def save_score(self):
        scores={"loss":self.score[0],'accuracy':self.score[1]}
        save_json(path=Path('score.json'),data=scores)
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            if tracking_url_type_store != "file":

                
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")

