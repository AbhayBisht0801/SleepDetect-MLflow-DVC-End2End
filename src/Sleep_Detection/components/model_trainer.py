from src.Sleep_Detection import logger
from src.Sleep_Detection.config.configuration import TraningConfig
import tensorflow as tf
from pathlib import Path
import os

class Training:
    def __init__(self,config:TraningConfig):
        self.config=config
    def get_base_model(self):
        self.model=tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    def train_valid_generator(self):
        datagenerator=dict(rescale=1./255,
                           validation_split=0.2)
        dataflow=dict(target_size=self.config.params_image_size[:-1],
                      batch_size=self.config.params_batch_size,
                      interpolation='bilinear')
        validation_datagenrator=tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator
        )
        self.valid_generator=validation_datagenrator.flow_from_directory(
            directory=self.config.training_data,
            subset='validation',
            shuffle=False,
            **dataflow
        )
        if self.config.params_is_augmentation:
            train_datagenerator=tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=90,
              
                zoom_range=0.2,
                horizontal_flip=True,
                **datagenerator
            )
        else:
            train_datagenerator=validation_datagenrator
        self.train_generator=train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset='training',
            shuffle=True,
            **dataflow
        )
    @staticmethod
    def save_model(path:Path,model:tf.keras.Model):
        model.save(path)

    def train(self):
        self.step_per_epoch=self.train_generator.samples//self.train_generator.batch_size
        self.validation_epoch=self.valid_generator.samples//self.valid_generator.batch_size
        self.model.fit(self.train_generator,epochs=self.config.params_epochs,
        steps_per_epoch=self.step_per_epoch,
        validation_steps=self.validation_epoch,
        validation_data=self.valid_generator)

        self.save_model(path=self.config.trained_model_path,model=self.model)