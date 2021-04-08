from keras.models import model_from_json

class Model:

    def __init__(self):
        self.audio_model = self.load_model()

    def load_model(self):
        loaded_model = model_from_json(self.read_json_file())
        loaded_model.load_weights('model_weights.h5')
        return loaded_model

    def read_json_file(self):
        file = open('model_file.json','r')
        model_json = file.read()
        return model_json

    def get_model(self):
        return self.audio_model
