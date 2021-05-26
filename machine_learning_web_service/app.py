import pickle
import numpy as np
import os
import json
import tornado.ioloop
import tornado.web

if not os.path.exists('my_model.pkl'):
    exit("Can't run without the model!")

with open('my_model.pkl', 'rb') as file:
    model = pickle.load(file)

class MainHandler(tornado.web.RequestHandler):
    # get and post are called when a get and post function is forwarded to the handler
    # just prints hello world when we go to the root path
    # demonstrates a get request
    def get(self):
        self.write("Hello, Tornado!")

class PredictionHandler(tornado.web.RequestHandler):
    # predicts one sample at a time
    def post(self):
        parameters = self.request.arguments
        x = np.array( list( map( float, parameters['input'] ) ) ) # data in http is passed as a string, so we convert to floats
        y = model.predict([x])[0]
        self.write(json.dumps({'prediction': y.item()}))
        self.finish()


if __name__ == "__main__":
    application = tornado.web.Application([
        (r"/", MainHandler),
        (r"/predict", PredictionHandler),
    ])
    application.listen(8888)
    tornado.ioloop.IOLoop.current().start()
