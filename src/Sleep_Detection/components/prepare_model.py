from src.Sleep_Detection.config.configuration import PrepareBaseModelConfig
from src.Sleep_Detection.utils.common import read_yaml,create_directories
from pathlib import Path
import tensorflow as tf



class PrepareBaseModel:
    def __init__(self,config:PrepareBaseModelConfig):
        self.config=config

    def get_base_model(self):
        self.model=tf.keras.applications.vgg16.VGG16(include_top=self.config.params_include_top,
                                                     weights=self.config.params_weights,
                                                     input_shape=self.config.params_image_size)
        
        model=self.model 
        model.summary()                                           
        self.save_model(path=self.config.base_model_path,model=self.model)

    @staticmethod
    def save_model(path:Path,model:tf.keras.models):
        model.save(path)
    def prepare_full_model(self,model,classes,frozen_all,frozen_till,learning_rate,dropout_no):
        if frozen_all:
            for layer in model.layers:
                model.trainable=False
        elif (frozen_till is not None) and (frozen_till>0):
            for layer in model.layers[:frozen_till]:
                model.trainable=False
        flatten_in=tf.keras.layers.Flatten()(model.output)
        dense1=tf.keras.layers.Dense(units=32,activation='relu')(flatten_in)
        dropout=tf.keras.layers.Dropout(dropout_no)(dense1)
        dense2=tf.keras.layers.Dense(units=16,activation='relu')(dropout)
        prediction=tf.keras.layers.Dense(units=classes,activation='softmax')(dense2)
        full_model=tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction

        )
        full_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=['accuracy'])
        return full_model
        
    def update_base_model(self):
        self.full_model=self.prepare_full_model(
                model=self.model,
                classes=self.config.params_classes,
                frozen_all=True,
                frozen_till=None,
                learning_rate=self.config.params_learning_rate,
                dropout_no=self.config.params_dropout
            )
        self.save_model(path=self.config.updated_base_model_path,model=self.full_model)


    @staticmethod
    def save_model(path:Path,model:tf.keras.Model):
        model.save(path)
        

