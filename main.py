from src.Sleep_Detection import logger
from src.Sleep_Detection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.Sleep_Detection.pipeline.stage_02_prepare_model import PrepareModelPipeline
from src.Sleep_Detection.pipeline.stage_03_model_trainer import ModelTrainingPipeline

STAGE_NAME='Data Ingestion Stage'

if __name__=='__main__':
    try:
        logger.info(f'>>>>>stage {STAGE_NAME} has Started')
        obj=DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f'>>>>>stage {STAGE_NAME} has completed')
    except Exception as e:
        logger.exception(e)
        raise e
    
STAGE_NAME='Prepare_Model'
if __name__=='__main__':
    try:
        logger.info(f'>>>>>stage {STAGE_NAME} has Started')
        obj=PrepareModelPipeline()
        obj.main()
        logger.info(f'>>>>>stage {STAGE_NAME} has completed')
    except Exception as e:
        logger.exception(e)
        raise e
STAGE_NAME='Model_Training'
if __name__=='__main__':
    try:
        logger.info(f'>>>>>stage {STAGE_NAME} has Started')
        obj=ModelTrainingPipeline()
        obj.main()
        logger.info(f'>>>>>stage {STAGE_NAME} has completed')
    except Exception as e:
        logger.exception(e)
        raise e