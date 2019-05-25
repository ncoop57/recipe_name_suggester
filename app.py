from flask import Flask
from PIL import Image
import recipe

app = Flask(__name__)
recipe_namer = recipe.get_model()


@app.route('/')
def hello():
    recipe_name = recipe_namer.predict(Image.open('./temp_image.jpg'))
    return ' '.join(recipe_name)


if __name__ == '__main__':
    app.run()
