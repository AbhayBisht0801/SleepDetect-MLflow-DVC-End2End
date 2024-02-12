from src.Sleep_Detection.config.configuration import ConfigurationManager
from src.Sleep_Detection.components.model_evaluation import Evaluation
from src.Sleep_Detection import logger

STAGE_NAME='Model_Evaluation'

class ModelEvaluation:
    def __init__(self):
        pass
    def main(self):
        config=ConfigurationManager()
        eval_config=config.get_evaluation_config()
        evaluation=Evaluation(config=eval_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()
if __name__=='__main__':
    try:
        logger.info(f'>>>>>stage {STAGE_NAME} has Started')
        obj=ModelEvaluation()
        obj.main()
        logger.info(f'>>>>>stage {STAGE_NAME} has completed')
    except Exception as e:
        logger.exception(e)
        raise e
