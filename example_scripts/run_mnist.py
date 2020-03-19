import sys
sys.path.append("..")
import keras, logging, random, pydot, copy, uuid, os, csv, json
from create_population import Create_Population
from Setup_ import Setup
import matplotlib.pyplot as plt
from enum import auto
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import regularizers
import imp
from keras.datasets import mnist
kerascodeepneat = imp.load_source("kerascodeepneat", "/home/daniel/github/Keras-CoDeepNEAT/base/kerascodeepneat.py")


def Log_Progress(filename='execution.log' ):

    logging.basicConfig(filename=filename,
                        filemode='w+', level=logging.INFO,
                        format='%(levelname)s - %(asctime)s: %(message)s')
    logging.addLevelName(21, "TOPOLOGY")
    logging.warning('This will get logged to a file')
    logging.info(f"Hi, this is a test run.")


def create_dir(dir):
    if not os.path.exists(os.path.dirname(dir)):
        try:
            os.makedirs(os.path.dirname(dir))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


if __name__ == "__main__":

    generations = 2
    training_epochs = 2
    final_model_training_epochs = 2
    population_size = 1
    blueprint_population_size = 10
    module_population_size = 30
    n_blueprint_species = 3
    n_module_species = 3

    create_dir("models/")
    create_dir("images/")

    setup = Setup(generations=2, 
                                training_epochs=2, 
                                population_size=1, 
                                blueprint_population_size=10, 
                                module_population_size=30, 
                                n_blueprint_species=3,
                                n_module_species=3)


    x_train, y_train, x_test, y_test, validation_split, batch_size, datagen = setup.dataset(mnist.load_data())
    compiler = setup.compiler_(loss= "categorical_crossentropy", optimizer= "keras.optimizers.Adam", lr=0.005, metrics = "accuracy")




    my_dataset = kerascodeepneat.Datasets(training=[x_train, y_train], test=[x_test, y_test])
    my_dataset.SAMPLE_SIZE = 10000
    my_dataset.TEST_SAMPLE_SIZE = 1000
    Log_Progress(filename = "Progress.log")



    es = EarlyStopping(monitor='val_acc', mode='auto', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model_checkpoint.h5', monitor='val_accuracy', mode='auto', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('training.csv')


    custom_fit_args = {"generator": datagen.flow(x_train, y_train, 
                        batch_size=batch_size),
                        "steps_per_epoch": x_train.shape[0] // batch_size,
                        "epochs": training_epochs,
                        "verbose": 1,
                        "validation_data": (x_test,y_test),
                        "callbacks": [es, csv_logger]
}       

    improved_dataset = kerascodeepneat.Datasets(training=[x_train, y_train], test=[x_test, y_test])
    improved_dataset.custom_fit_args = custom_fit_args
    my_dataset.custom_fit_args = None


    population = kerascodeepneat.Population(my_dataset, input_shape=x_train.shape[1:], population_size=population_size, compiler=compiler)
    create = Create_Population(population)

    create.modules(module_population_size, n_module_species)
    create.blueprints(blueprint_population_size, n_blueprint_species)

    iteration = population.iterate_generations(generations=generations,
                                                training_epochs=training_epochs,
                                                validation_split=validation_split,
                                                mutation_rate=0.5,
                                                crossover_rate=0.2,
                                                elitism_rate=0.1,
                                                possible_components=create.possible_components,
                                                possible_complementary_components=create.possible_complementary_components)


    print("Best fitting: (Individual name, Blueprint mark, Scores[test loss, test acc], History).\n", (iteration))

     # # Return the best model
    best_model = population.return_best_individual()

    # Set data augmentation for full training
    population.datasets = improved_dataset
    print("Using data augmentation.")

    try:
        print("Best fitting model chosen for retraining: {best_model.name}")
        population.train_full_model(best_model, final_model_training_epochs, validation_split, custom_fit_args)
    except:
        population.individuals.remove(best_model)
        best_model = population.return_best_individual()
        print("Best fitting model chosen for retraining: {best_model.name}")
        population.train_full_model(best_model, final_model_training_epochs, validation_split, custom_fit_args)


  