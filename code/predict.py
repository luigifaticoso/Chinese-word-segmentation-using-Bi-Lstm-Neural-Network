from argparse import ArgumentParser
import importlib


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path",default='public_homework_1/resources/output.utf8' ,help="The path of the input file")
    parser.add_argument("output_path",default='public_homework_1/resources/output.txt' ,help="The path of the output file")
    parser.add_argument("resources_path",default='public_homework_1/resources', help="The path of the resources needed to load your model")

    return parser.parse_args()


def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    train_x,train_y,test_x,test_y,dev_x,dev_y,vocabolary = make_dataset(input_path,'dataset.txt')

    model = create_model(len(vocabolary,20))
    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=4,
            validation_data=[x_test, y_test])
    pass


if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
