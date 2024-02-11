from src.Sleep_Detection.config.configuration import ConfigurationManager
from src.Sleep_Detection.components.model_trainer import Training
from src.Sleep_Detection import logger


class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config=ConfigurationManager()
        trainer_config=config.get_training_config()
        training=Training(config=trainer_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()
STAGE_NAME='Model training'
    
if __name__=='__main__':
    try:
        logger.info(f'>>>>>stage {STAGE_NAME} has Started')
        obj=ModelTrainingPipeline()
        obj.main()
        logger.info(f'>>>>>stage {STAGE_NAME} has completed')
    except Exception as e:
        logger.exception(e)
        raise e