import pickle


def get_data_from_pickle(file_path):
    """
    Charge un fichier pickle


    Entrées

        file_path - str : Chemin vers le fichier pickle


    Sortie

        data            : Contenu du fichier pickle
    """

    with open(file_path, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def save_data_to_pickle(data, file_path):
    """
    Sauvegarde des données dans un fichier pickle


    Entrées

        data            : Données à sauvegarder
        file_path - str : Chemin vers le fichier de sortie
    """

    with open(file_path, 'wb') as fo:
        pickle.dump(data, fo)