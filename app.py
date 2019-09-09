from flask import Flask, request
from PIL import Image
import recipe

app = Flask(__name__)
recipe_namer = recipe.get_model()


@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        recipe_name = recipe_namer.predict(Image.open(request.files['file']))
        print("Got Post request")
        return ' '.join(recipe_name)
    if request.method == 'GET':
        recipe_name = recipe_namer.predict(Image.open('/workspaces/recipe_name_suggester/temp_image.jpg'))
        return ' '.join(recipe_name)


if __name__ == '__main__':
    app.run()
