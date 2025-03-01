import pickle

def load_data_from_pickle_file(path_to_file: str):
    """
    Charge un fichier pickle.

    Entrée
    ------
    path_to_file - str
        Chemin vers le fichier pickle.

    Sortie
    ------
    data
        Contenu du fichier pickle.
    """
    with open(path_to_file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def save_data_to_pickle_file(data, path_to_file: str):
    """
    Sauvegarde des données dans un fichier pickle.

    Entrées
    -------
    data
        Données à sauvegarder.

    path_to_file - str
        Chemin vers le fichier de sortie.
    """
    with open(path_to_file, 'wb') as fo:
        pickle.dump(data, fo)
