from src.Sleep_Detection.config.configuration import ConfigurationManager
from src.Sleep_Detection.components.prepare_model import PrepareBaseModel
from src.Sleep_Detection import logger



STAGE_NAME='Prepare Base Model'


class PrepareModel:
    def __init__(self):
        pass
    def main(self):
        config=ConfigurationManager()
        prepare_base_model_config=config.prepare_base_model()
        prepare_base_model=PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


if __name__=='__main__':
    try:
        logger.info(f'>>>>>stage {STAGE_NAME} has Started')
        obj=PrepareModel()
        obj.main()
        logger.info(f'>>>>>stage {STAGE_NAME} has completed')
    except Exception as e:
        logger.exception(e)
        raise e

        
            
        
